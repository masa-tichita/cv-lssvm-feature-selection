"""Model spec of SVM-CV, the cross-validation-based MIO formulation of Purcaru et al. (2026).

SVM-CV is evaluated by the features selected by its MIO solution; test
predictions use the mean decision value of the per-fold models (w^(k), b^(k))
of the same solution.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from modules.algo.svm_cv.solver import SVMCVSolver
from modules.algo.svm_cv.types import SVMCVParams, SVMCVSolution
from modules.models.common import (
    evaluate_mio_solution,
    gurobi_log_dir,
    load_fold_models,
    mio_config_kwargs,
    mio_meta,
    mio_solution_fields,
    mio_train_metrics,
    save_fold_models,
    select_min_mean_objective,
    selection_meta,
)
from modules.experiment.layout import read_model_meta, write_model_meta
from modules.settings import AppSettings


class SVMCVSpec:
    """SVM-CV model spec."""

    name = "svm_cv"

    def __init__(self) -> None:
        self._solver = SVMCVSolver()

    def get_param_grid(self, settings: AppSettings) -> dict[str, list[Any]]:
        """Return the search space of C."""
        return {"C": settings.models.svm_cv.search.C_list}

    def train(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        params: dict[str, Any],
        settings: AppSettings,
    ) -> tuple[SVMCVSolution, float]:
        """Select features by solving the SVM-CV MIO for the given C."""
        config = settings.models.svm_cv.config
        svm_cv_params = SVMCVParams(
            C=float(params["C"]),
            n_folds=config.n_folds,
            shuffle=config.shuffle,
            seed=config.cv_seed,
            optimality_tol=config.optimality_tol,
            log_dir=gurobi_log_dir(params, self.name),
            **mio_config_kwargs(config),
        )
        solution = self._solver.train(X_train, y_train, svm_cv_params)
        return solution, solution.obj_value

    def extract_train_metrics(self, solution: SVMCVSolution) -> dict[str, Any]:
        """Extract search metrics from the training result."""
        return mio_train_metrics(solution)

    def select_best(self, results_df: pd.DataFrame) -> dict[float, dict[str, Any]]:
        """Select, for each SNR, the C with the smallest SVM-CV objective (ties: smaller C)."""
        return select_min_mean_objective(results_df, "C")

    def save(self, solution: SVMCVSolution, path: Path) -> None:
        """Save the trained model (meta.json; model.npz with z and the fold models)."""
        meta = {
            **selection_meta(self.name, solution),
            "C": solution.C,
            **mio_meta(solution),
        }
        write_model_meta(path, meta)
        save_fold_models(path, solution.z, solution.fold_models)

    def load(self, path: Path) -> SVMCVSolution:
        """Load a trained model."""
        meta = read_model_meta(path)
        z, fold_models = load_fold_models(path)
        return SVMCVSolution(
            z=z,
            C=meta["C"],
            fold_models=fold_models,
            **mio_solution_fields(meta),
        )

    def predict(
        self,
        solution: SVMCVSolution,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict with the mean decision value of the per-fold models."""
        predictions, _ = self._solver.predict(X, solution)
        return predictions

    def evaluate(
        self,
        solution: SVMCVSolution,
        X_test: NDArray[np.float64],
        y_test: NDArray[np.float64],
        true_support: list[int] | None,
        settings: AppSettings,
    ) -> dict[str, Any]:
        """Evaluate the per-fold models of the MIO solution on the test data."""
        predictions, decision_values = self._solver.predict(X_test, solution)
        return evaluate_mio_solution(
            solution, y_test, predictions, decision_values, true_support
        )
