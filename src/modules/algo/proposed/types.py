from typing import Literal

from pydantic import Field, model_validator

from modules.algo.common.types import FoldModel, MIOParams, MIOSolution

# Validation model of the proposed method:
#   "hl": hinge-loss-based validation model (Proposed-HL, a MILO problem)
#   "ls": least-squares-based validation model (Proposed-LS, a MIQO problem)
Variant = Literal["hl", "ls"]


class ProposedParams(MIOParams):
    """Parameters of the proposed method (used for the actual run).

    The Gurobi log is written to ``log_dir/gamma={gamma}.log`` when ``log_dir`` is set.
    """

    gamma: float = Field(gt=0.0)
    variant: Variant = Field(default="hl")
    n_folds: int = Field(default=5, ge=2)
    shuffle: bool = Field(default=True)
    seed: int | None = None
    # Bounds on continuous variables (for tightening Big-M)
    var_bound_w: float | None = Field(default=None, gt=0.0)
    var_bound_b: float | None = Field(default=None, gt=0.0)
    # Add Big-M linking constraints to tighten the LP relaxation
    bigm_link: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_bigm_link(self) -> "ProposedParams":
        if self.bigm_link and self.var_bound_w is None:
            raise ValueError("bigm_link=True requires var_bound_w to be set")
        return self


class ProposedSolution(MIOSolution):
    """Optimization result of the proposed method.

    Each of the K folds has its own model (w^(k), b^(k)).
    Prediction uses the mean decision value over all folds.
    """

    fold_models: list[FoldModel]
