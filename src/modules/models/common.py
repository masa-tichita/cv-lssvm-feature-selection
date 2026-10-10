"""Helpers shared by the model specs.

* Gurobi settings and log directory of the MIO methods (proposed, SVM-CV, LogReg).
* Search metrics, evaluation rows, and parameter selection.
* Saving and loading of trained models (``meta.json``, ``model.npz``).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from modules.algo.common.types import FoldModel, MIOSolution
from modules.evaluation.metrics import compute_evaluation_metrics
from modules.experiment.layout import ResultLayout
from modules.settings import LogRegConfig, ProposedConfig, SVMCVConfig
from utils.gurobi_resources import gurobi_status_name

META_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Training (MIO methods)
# ---------------------------------------------------------------------------


def mio_config_kwargs(
    config: ProposedConfig | SVMCVConfig | LogRegConfig,
) -> dict[str, Any]:
    """Solver settings shared by all MIO methods (fields of ``MIOParams``)."""
    return {
        "max_features": config.max_features,
        "time_limit": config.time_limit,
        "opt_gap": config.opt_gap,
        "threads": config.threads,
        "nodefile_start": config.nodefile_start,
        "soft_mem_limit": config.soft_mem_limit,
        "nodefile_dir": config.nodefile_dir,
    }


def gurobi_log_dir(params: dict[str, Any], model_name: str) -> Path | None:
    """Gurobi log directory from the run context the runner adds to ``params``.

    Returns ``<run_dir>/training/gurobi_logs/<model>/snr=<snr>/seed=<seed>``,
    or None (no log file) when the context is missing.
    """
    run_dir = params.get("run_dir")
    snr = params.get("snr")
    seed = params.get("seed")
    if run_dir is None or snr is None or seed is None:
        return None
    return ResultLayout(Path(run_dir)).gurobi_log_dir(model_name, float(snr), int(seed))


# ---------------------------------------------------------------------------
# Search metrics and evaluation rows
# ---------------------------------------------------------------------------


def _gurobi_fields(solution: MIOSolution) -> dict[str, Any]:
    return {
        "opt_gap": solution.opt_gap,
        "solver_status": solution.solver_status,
        "solver_status_name": gurobi_status_name(solution.solver_status),
    }


def mio_train_metrics(solution: MIOSolution) -> dict[str, Any]:
    """Search metrics of an MIO solution."""
    return {
        "n_selected": solution.n_selected_features,
        "solve_time": solution.solve_time,
        **_gurobi_fields(solution),
    }


def cv_score_train_metrics(solution: Any) -> dict[str, Any]:
    """Search metrics of a CV-scored sklearn selection (SVM-L1, SVM-RFE)."""
    return {
        "n_selected": solution.n_selected_features,
        "solve_time": solution.solve_time,
        "cv_score": solution.cv_score,
    }


def evaluation_row(
    objective: float,
    solution: Any,
    solver_fields: dict[str, Any],
    y_test: Any,
    predictions: Any,
    decision_values: Any,
    true_support: list[int] | None,
) -> dict[str, Any]:
    """One row of eval_results.csv: objective, subset, solver fields, then the metrics."""
    return {
        "objective": objective,
        "n_selected": solution.n_selected_features,
        "selected_features": solution.selected_features,
        "solve_time": solution.solve_time,
        **solver_fields,
        **compute_evaluation_metrics(
            y_test,
            predictions,
            decision_values,
            solution.selected_features,
            true_support,
        ),
    }


def evaluate_mio_solution(
    solution: MIOSolution,
    y_test: Any,
    predictions: Any,
    decision_values: Any,
    true_support: list[int] | None,
) -> dict[str, Any]:
    """Evaluation row of an MIO solution (objective = MIO objective value)."""
    return evaluation_row(
        solution.obj_value,
        solution,
        _gurobi_fields(solution),
        y_test,
        predictions,
        decision_values,
        true_support,
    )


def evaluate_cv_score_solution(
    solution: Any,
    y_test: Any,
    predictions: Any,
    decision_values: Any,
    true_support: list[int] | None,
) -> dict[str, Any]:
    """Evaluation row of a CV-scored sklearn selection (objective = -CV score)."""
    return evaluation_row(
        -solution.cv_score,
        solution,
        {},
        y_test,
        predictions,
        decision_values,
        true_support,
    )


# ---------------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------------


def select_min_mean_objective(
    results_df: pd.DataFrame,
    param: str,
) -> dict[float, dict[str, Any]]:
    """For each SNR, the value of ``param`` with the smallest mean objective.

    Ties are broken toward the smaller parameter value.
    """
    best_params: dict[float, dict[str, Any]] = {}
    for snr in results_df["snr"].unique():
        snr_df = results_df[results_df["snr"] == snr]
        means = snr_df.groupby(param)["objective"].mean()
        min_obj = float(means.min())
        best = min(float(v) for v in means.index if means[v] == min_obj)
        best_params[float(snr)] = {param: best}
        logger.info(f"SNR={snr}: best {param}={best:g} (mean objective={min_obj:.4f})")
    return best_params


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def selection_meta(model_name: str, solution: Any) -> dict[str, Any]:
    """Leading fields of meta.json: schema, model, and the selected subset."""
    return {
        "schema_version": META_SCHEMA_VERSION,
        "model_name": model_name,
        "n_selected": int(solution.n_selected_features),
        "selected_features": [int(f) for f in solution.selected_features],
    }


def mio_meta(solution: MIOSolution) -> dict[str, Any]:
    """Solver fields of meta.json for an MIO solution."""
    return {
        "obj_value": float(solution.obj_value),
        "opt_gap": float(solution.opt_gap) if solution.opt_gap is not None else None,
        "solve_time": float(solution.solve_time),
        "solver_status": solution.solver_status,
    }


def mio_solution_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments of an MIOSolution restored from meta.json (except z)."""
    return {
        "selected_features": meta["selected_features"],
        "n_selected_features": meta["n_selected"],
        "obj_value": meta["obj_value"],
        "opt_gap": meta.get("opt_gap"),
        "solve_time": meta["solve_time"],
        "solver_status": meta.get("solver_status"),
    }


def save_fold_models(
    path: Path,
    z: np.ndarray,
    fold_models: list[FoldModel],
) -> None:
    """Write model.npz with z and the per-fold models (fold_w: K x p, fold_b: K)."""
    fold_w = np.array([fm.w for fm in fold_models])
    fold_b = np.array([fm.b for fm in fold_models])
    np.savez(path / "model.npz", z=z, fold_w=fold_w, fold_b=fold_b)


def load_fold_models(path: Path) -> tuple[np.ndarray, list[FoldModel]]:
    """Read z and the per-fold models written by ``save_fold_models``."""
    data = np.load(path / "model.npz")
    fold_w = data["fold_w"]
    fold_b = data["fold_b"]
    fold_models = [
        FoldModel(w=fold_w[k], b=float(fold_b[k])) for k in range(len(fold_b))
    ]
    return data["z"], fold_models
