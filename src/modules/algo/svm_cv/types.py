from pydantic import Field, model_validator

from modules.algo.common.types import FoldModel, MIOParams, MIOSolution


class SVMCVParams(MIOParams):
    """Parameters of SVM-CV (Purcaru et al., 2026).

    The Gurobi log is written to ``log_dir/svm_cv_C=<C>_<timestamp>.log`` when
    ``log_dir`` is set.
    """

    C: float = Field(gt=0.0)
    n_folds: int = Field(default=10, ge=2)
    shuffle: bool = Field(default=True)
    seed: int | None = None
    optimality_tol: float | None = Field(default=1e-7, gt=0.0)


class SVMCVSolution(MIOSolution):
    """Solution of SVM-CV: selected features and per-fold models of the MIO."""

    C: float = Field(gt=0.0)
    fold_models: list[FoldModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_fold_shapes(self) -> "SVMCVSolution":
        for fold_model in self.fold_models:
            if fold_model.w.shape != self.z.shape:
                raise ValueError("fold w and z shape mismatch")
        return self
