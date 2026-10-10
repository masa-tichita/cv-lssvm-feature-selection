"""Common L2-SVM refit used for the AUC values reported in the paper.

For every method, trial (SNR, seed), and saved feature subset S, a linear
L2-SVM is refitted on the training data restricted to S:

    minimize 0.5 * ||w_S||^2 + C * sum_i xi_i
    s.t.     y_i (w_S^T x_iS + b) >= 1 - xi_i,  xi_i >= 0.

C is not divided by the sample size. C is chosen from the configured grid
(``models.refit_svm.search.C_list``) by stratified K-fold CV on the training
data (``models.refit_svm.config``: K=5, shuffle=True, seed=42 in the paper)
maximizing the mean validation accuracy; ties are broken toward the smaller C.
The test AUC is computed from the decision values X_test @ w + b of this refitted SVM.

Feature selection itself is not rerun: the subsets are read from
``training/models/**/meta.json`` of an experiment result directory.
"""

import ast
import time
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from loguru import logger
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score

from modules.data.factory import DataConfig, build_run_data_config, create_dataset
from modules.data.splits import split_k_fold_indices
from modules.evaluation.metrics import compute_classification_metrics
from modules.experiment.layout import ResultLayout, read_model_meta
from modules.settings import RefitSVMSettings
from utils.labels import METHOD_LABELS
from utils.ml import linear_decision, normalize_binary_labels, predict_labels


def fit_l2_svm_sum(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    support: list[int],
    C: float,
    opt_gap: float = 1e-4,
    time_limit: float | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Solve 0.5 ||w_S||^2 + C sum xi s.t. y_i (w_S^T x_iS + b) >= 1 - xi_i, xi >= 0."""
    y = normalize_binary_labels(y)
    n, p = X.shape
    mdl = gp.Model("l2_svm_refit_sumC")
    mdl.Params.OutputFlag = 0
    w_sel = mdl.addVars(len(support), lb=-GRB.INFINITY, name="w")
    b = mdl.addVar(lb=-GRB.INFINITY, name="b")
    xi = mdl.addVars(n, lb=0.0, name="xi")
    mdl.setObjective(
        0.5 * gp.quicksum(w_sel[k] * w_sel[k] for k in range(len(support)))
        + C * gp.quicksum(xi[i] for i in range(n)),
        GRB.MINIMIZE,
    )
    for i in range(n):
        margin = (
            gp.quicksum(w_sel[k] * float(X[i, j]) for k, j in enumerate(support)) + b
        )
        mdl.addConstr(float(y[i]) * margin >= 1.0 - xi[i])
    mdl.Params.MIPGap = opt_gap
    if time_limit is not None:
        mdl.Params.TimeLimit = time_limit
    mdl.optimize()
    if mdl.Status != GRB.OPTIMAL:
        status = mdl.Status
        mdl.dispose()
        raise RuntimeError(f"L2-SVM refit did not reach optimality (status={status})")
    w = np.zeros(p)
    for k, j in enumerate(support):
        w[j] = w_sel[k].X
    bv = float(b.X)
    mdl.dispose()
    return w, bv


def refit_sum(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    support: list[int],
    settings: RefitSVMSettings,
) -> tuple[NDArray[np.float64], float, float, float, list[dict[str, float]]]:
    """Select C by stratified K-fold CV (accuracy, ties -> smaller C) and refit."""
    config = settings.config
    C_list = settings.search.C_list
    y = normalize_binary_labels(y_train)
    tr, va = split_k_fold_indices(
        X_train.shape[0], y, config.n_folds, config.shuffle, config.cv_seed
    )
    best_C, best_score = float(C_list[0]), -np.inf
    cv_results: list[dict[str, float]] = []
    for C in C_list:
        scores = []
        for t, v in zip(tr, va, strict=True):
            w, b = fit_l2_svm_sum(
                X_train[t], y[t], support, float(C), config.opt_gap, config.time_limit
            )
            pred = predict_labels(linear_decision(X_train[v], w, b))
            scores.append(float(accuracy_score(y[v], pred)))
        score = float(np.mean(scores))
        cv_results.append({"C": float(C), "cv_score": score})
        if score > best_score or (score == best_score and float(C) < best_C):
            best_C, best_score = float(C), score
    w, b = fit_l2_svm_sum(
        X_train, y, support, best_C, config.opt_gap, config.time_limit
    )
    return w, b, best_C, best_score, cv_results


class _NumpyInts(ast.NodeTransformer):
    """Strip ``np.int64(...)`` wrappers from a saved ``selected_features`` string."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in ("int64", "int32")
            and len(node.args) == 1
            and not node.keywords
        ):
            return self.visit(node.args[0])
        raise ValueError("Unexpected expression in saved support")


def _support_from_csv(text: str) -> list[int]:
    tree = _NumpyInts().visit(ast.parse(text, mode="eval"))
    return [int(j) for j in ast.literal_eval(tree)]


def collect_supports(result_dir: Path) -> list[dict[str, Any]]:
    """Read the saved feature subsets of every method and trial."""
    layout = ResultLayout(result_dir)
    config = layout.read_config()
    eval_df = pd.read_csv(layout.eval_results_csv)
    records: list[dict[str, Any]] = []
    for path in sorted(layout.models_dir.glob("**/meta.json")):
        model, snr, seed = layout.parse_model_dir(path)
        meta = read_model_meta(path.parent)
        support = [int(j) for j in meta["selected_features"]]
        row = eval_df[
            (eval_df["model"] == model)
            & (eval_df["snr"] == snr)
            & (eval_df["seed"] == seed)
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one eval row for {model}, {snr}, {seed}")
        if support != _support_from_csv(str(row.iloc[0]["selected_features"])):
            raise ValueError(f"Saved support mismatch: {path}")
        if not all(0 <= j < config["data"]["features"] for j in support):
            raise ValueError(f"Feature index out of range: {path}")
        records.append({"model": model, "snr": snr, "seed": seed, "support": support})
    return records


def refit_protocol(settings: RefitSVMSettings, run_count: int) -> dict[str, Any]:
    """Describe the refit protocol (written to ``protocol.json``)."""
    return {
        "selection": "saved support, unchanged",
        "loss": "hinge",
        "regularization": "L2",
        "objective": "0.5 ||w||^2 + C sum(hinge)  (C not divided by n)",
        "C_list": settings.search.C_list,
        "n_folds": settings.config.n_folds,
        "shuffle": settings.config.shuffle,
        "cv_seed": settings.config.cv_seed,
        "C_selection_metric": "accuracy",
        "C_tie_break": "smaller C",
        "standardization": "none",
        "run_count": run_count,
    }


def refit_supports(
    records: list[dict[str, Any]],
    data_config: dict[str, Any],
    settings: RefitSVMSettings,
) -> pd.DataFrame:
    """Refit the L2-SVM on every saved subset and evaluate it on the test data.

    Returns one row per (method, SNR, seed) with the selected C, its CV accuracy,
    and the test AUC / accuracy of the refitted SVM. The test AUC must be
    defined (both classes in the test data); otherwise ValueError is raised.
    """
    base_config = DataConfig.model_validate(data_config)
    records = sorted(records, key=lambda r: (r["snr"], r["seed"]))
    rows: list[dict[str, Any]] = []
    data_key: tuple[float, int] | None = None
    data = None
    start = time.perf_counter()
    for idx, rec in enumerate(records):
        model, snr, seed, support = rec["model"], rec["snr"], rec["seed"], rec["support"]
        if (snr, seed) != data_key:
            data = create_dataset(build_run_data_config(base_config, snr=snr, seed=seed))
            data_key = (snr, seed)
        assert data is not None
        w, b, C, cv, _ = refit_sum(data.X_train, data.y_train, support, settings)
        dec = linear_decision(data.X_test, w, b)
        test = compute_classification_metrics(data.y_test, predict_labels(dec), dec)
        if test["model_auc"] is None:
            raise ValueError(f"Test AUC undefined for {model}, snr={snr}, seed={seed}")
        rows.append(
            {
                "model": model,
                "method": METHOD_LABELS.get(model, model),
                "snr": snr,
                "seed": seed,
                "support_size": len(support),
                "final_C": C,
                "cv_accuracy": cv,
                "refit_auc": test["model_auc"],
                "refit_accuracy": test["model_accuracy"],
            }
        )
        if (idx + 1) % 200 == 0:
            logger.info(
                f"{idx + 1}/{len(records)} elapsed={time.perf_counter() - start:.1f}s"
            )
    return pd.DataFrame(rows)


def summarize_refit(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard error of the refit test AUC per method and SNR."""
    return (
        frame.groupby(["model", "method", "snr"], sort=False)
        .agg(
            trials=("refit_auc", "size"),
            refit_auc=("refit_auc", "mean"),
            refit_auc_se=("refit_auc", lambda s: s.std() / len(s) ** 0.5),
        )
        .reset_index()
    )
