"""Gurobi handling and prediction shared by the MIO solvers.

The formulations themselves stay in each method's solver; this module only
holds the steps that are the same for all of them: solver settings and log
file, status check after optimization, extraction of the solution, and
prediction with the per-fold models.
"""

from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from loguru import logger
from numpy.typing import NDArray

from modules.algo.common.types import FoldModel, MIOParams
from utils.gurobi_resources import configure_gurobi_resources
from utils.ml import linear_decision

# Statuses accepted when Gurobi stopped early but found a feasible solution
STOPPED_WITH_SOLUTION: frozenset[int] = frozenset(
    {GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.MEM_LIMIT}
)


def gurobi_log_file(log_dir: Path | None, file_name: str) -> str:
    """Return the log file path in ``log_dir`` (created), or "" (no log file) if None."""
    if log_dir is None:
        return ""
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / file_name)


def configure_mio_model(model: gp.Model, params: MIOParams, log_path: str) -> None:
    """Apply the shared Gurobi settings: log file only, MIP gap, time limit, resources."""
    model.Params.OutputFlag = 1
    model.Params.LogFile = log_path
    model.Params.LogToConsole = 0
    model.Params.MIPGap = params.opt_gap
    if params.time_limit is not None:
        model.Params.TimeLimit = params.time_limit
    configure_gurobi_resources(
        model,
        threads=params.threads,
        nodefile_start=params.nodefile_start,
        soft_mem_limit=params.soft_mem_limit,
        nodefile_dir=params.nodefile_dir,
    )


def optimize_mio_model(
    model: gp.Model,
    label: str,
    accepted_statuses: frozenset[int] = STOPPED_WITH_SOLUTION,
) -> None:
    """Optimize and require a usable solution.

    OPTIMAL is always accepted; a status in ``accepted_statuses`` is accepted
    (with a warning) when a feasible solution exists. Otherwise the model is
    disposed of and RuntimeError is raised.
    """
    model.optimize()
    status = model.Status
    if status == GRB.OPTIMAL:
        return
    if status in accepted_statuses and model.SolCount > 0:
        logger.warning(
            f"{label} optimization stopped with status={status} "
            "but a feasible solution was found"
        )
        return
    model.dispose()
    raise RuntimeError(f"{label} optimization failed with status {status}")


def mip_gap(model: gp.Model) -> float | None:
    """Final relative MIP gap, or None if Gurobi reports no finite gap."""
    if model.MIPGap < GRB.INFINITY:
        return float(model.MIPGap)
    return None


def var_values(variables: gp.tupledict, p: int) -> NDArray[np.float64]:
    """Values of the variables indexed 0..p-1."""
    return np.array([variables[j].X for j in range(p)], dtype=np.float64)


def selected_from_z(z: NDArray[np.float64]) -> list[int]:
    """Indices of the selected features (z_j > 0.5)."""
    return [j for j in range(len(z)) if z[j] > 0.5]


def fold_models_from_vars(
    w_vars: gp.tupledict,
    b_vars: gp.tupledict,
    n_folds: int,
    p: int,
) -> list[FoldModel]:
    """Per-fold models (w^(k), b^(k)) from the variables w[k, j] and b[k]."""
    return [
        FoldModel(
            w=np.array([w_vars[k, j].X for j in range(p)], dtype=np.float64),
            b=float(b_vars[k].X),
        )
        for k in range(n_folds)
    ]


def log_solved(label: str, n_selected: int, p: int, obj_value: float, solve_time: float) -> None:
    logger.info(
        f"{label} solved: selected {n_selected}/{p} features, "
        f"obj={obj_value:.4f}, time={solve_time:.2f}s"
    )


def mean_fold_decision(
    X: NDArray[np.float64],
    fold_models: list[FoldModel],
) -> NDArray[np.float64]:
    """Mean decision value over the per-fold models: mean_k (X @ w^(k) + b^(k))."""
    all_decisions = np.zeros((len(fold_models), X.shape[0]), dtype=np.float64)
    for k, fold_model in enumerate(fold_models):
        all_decisions[k] = linear_decision(X, fold_model.w, fold_model.b)
    return np.mean(all_decisions, axis=0)
