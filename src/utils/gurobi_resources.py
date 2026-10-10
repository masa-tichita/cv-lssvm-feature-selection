"""Apply Gurobi CPU and memory safety settings consistently."""

from pathlib import Path
from tempfile import gettempdir

import gurobipy as gp
from gurobipy import GRB

_STATUS_NAMES = {
    GRB.OPTIMAL: "optimal",
    GRB.TIME_LIMIT: "time_limit",
    GRB.MEM_LIMIT: "memory_limit",
    GRB.INTERRUPTED: "interrupted",
    GRB.SUBOPTIMAL: "suboptimal",
    GRB.NUMERIC: "numeric",
}


def gurobi_status_name(status: int | None) -> str | None:
    """Map the main Gurobi status codes to names used in the analysis."""
    if status is None:
        return None
    return _STATUS_NAMES.get(status, f"status_{status}")


def configure_gurobi_resources(
    model: gp.Model,
    *,
    threads: int | None,
    nodefile_start: float | None,
    soft_mem_limit: float | None,
    nodefile_dir: str | None,
) -> None:
    """Apply only the given settings to a Gurobi model (memory in GB)."""
    if threads is not None:
        model.Params.Threads = threads
    if nodefile_start is not None:
        model.Params.NodefileStart = nodefile_start
    if soft_mem_limit is not None:
        model.Params.SoftMemLimit = soft_mem_limit
    if nodefile_start is not None or nodefile_dir is not None:
        directory = (
            Path(nodefile_dir).expanduser()
            if nodefile_dir is not None
            else Path(gettempdir()) / "cvlssvm-gurobi-nodefiles"
        ).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        model.Params.NodefileDir = str(directory)
