from pathlib import Path
from typing import Any

import gurobipy as gp

from utils.system import resolve_project_path


def save_gurobi_model(
    model: gp.Model,
    base_dir: Path | str,
    model_name: str = "proposed",
    **params: Any,
) -> dict[str, Path]:
    """Save a Gurobi model in LP/MPS format.

    Output: {project_root}/{base_dir}/{model_name}/{param1}_{value1}_{param2}_{value2}.../

    Args:
        model: Gurobi model.
        base_dir: Base directory (relative paths are resolved from project_root()).
        model_name: Model name (subdirectory name).
        **params: Additional parameters (included in the file name).

    Returns:
        {"lp": Path, "mps": Path}
    """
    base_path = resolve_project_path(base_dir)

    # Sort parameters alphabetically
    sorted_params = sorted(params.items())
    param_str = (
        "_".join(f"{k}_{v}" for k, v in sorted_params) if sorted_params else "default"
    )
    output_dir = base_path / model_name / param_str
    output_dir.mkdir(parents=True, exist_ok=True)

    lp_path = output_dir / "model.lp"
    mps_path = output_dir / "model.mps"

    model.write(str(lp_path))
    model.write(str(mps_path))

    return {"lp": lp_path, "mps": mps_path}
