"""Types shared by the MIO solvers (proposed method, SVM-CV, LogReg)."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MIOParams(BaseModel):
    """Solver settings shared by all MIO methods."""

    max_features: int | None = Field(default=None, ge=1)
    time_limit: float | None = Field(default=None, ge=0.0)
    opt_gap: float = Field(default=1e-4, ge=0.0)
    threads: int | None = Field(default=None, ge=1)
    nodefile_start: float | None = Field(default=None, ge=0.0)
    soft_mem_limit: float | None = Field(default=None, gt=0.0)
    nodefile_dir: str | None = None
    # Directory of the Gurobi log files (injected in experiment runs;
    # None for standalone calls, which write no log file)
    log_dir: Path | None = None

    model_config = ConfigDict(frozen=True)


class FoldModel(BaseModel):
    """Model (w^(k), b^(k)) obtained for one fold."""

    w: NDArray[np.float64]
    b: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MIOSolution(BaseModel):
    """Selected features and solver information of an MIO solution."""

    z: NDArray[np.float64]
    selected_features: list[int]
    n_selected_features: int = Field(ge=0)
    obj_value: float
    opt_gap: float | None = None
    solve_time: float = Field(ge=0.0)
    solver_status: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_consistency(self) -> "MIOSolution":
        if self.n_selected_features != len(self.selected_features):
            raise ValueError("n_selected_features mismatch")
        # Count z > 0.5 to allow for numerical error (Gurobi may return non-integral values)
        z_count = int(np.sum(self.z > 0.5))
        if z_count != self.n_selected_features:
            raise ValueError(
                f"z count ({z_count}) != n_selected_features ({self.n_selected_features})"
            )
        return self
