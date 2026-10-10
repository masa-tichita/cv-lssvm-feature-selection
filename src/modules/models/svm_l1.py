"""SVM-L1 model spec.

ModelSpec implementation of the L1-regularized (sparse) SVM.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from modules.models.common import (
    cv_score_train_metrics,
    evaluate_cv_score_solution,
    select_min_mean_objective,
    selection_meta,
)
from modules.experiment.layout import read_model_meta, write_model_meta
from modules.settings import AppSettings
from utils.ml import safe_n_folds


@dataclass
class SVML1Solution:
    """Training result of SVM-L1."""

    # Indices of the selected features (nonzero coefficients)
    selected_features: list[int]
    n_selected_features: int
    # Coefficient vector
    coef: NDArray[np.float64]
    # Trained SVM
    svm: LinearSVC
    # CV score (negated for minimization)
    cv_score: float
    # Training time
    solve_time: float


class SVML1Spec:
    """SVM-L1 model spec."""

    name = "svm_l1"

    def get_param_grid(self, settings: AppSettings) -> dict[str, list[Any]]:
        """Return the search space."""
        s = settings.models.svm_l1
        return {"C": s.search.C_list}

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        params: dict[str, Any],
        settings: AppSettings,
    ) -> tuple[SVML1Solution, float]:
        """Train the model.

        Returns:
            (solution, objective_value): objective is -cv_score (minimized).
        """
        s = settings.models.svm_l1
        config = s.config

        C = params["C"]

        start_time = perf_counter()

        # L1-regularized SVM (the L1 penalty is supported only in the primal)
        svm = LinearSVC(
            penalty="l1",
            loss="squared_hinge",
            C=C,
            dual=False,  # type: ignore[arg-type]
            random_state=config.cv_seed,
            max_iter=10000,
        )

        # Compute the CV score (number of folds adjusted to the class sample sizes)
        n_folds = safe_n_folds(y_train, config.n_folds)
        cv_scores = cross_val_score(
            svm, X_train, y_train, cv=n_folds, scoring="accuracy"
        )
        cv_score = float(cv_scores.mean())

        # Train the final model
        svm.fit(X_train, y_train)

        # Select features with nonzero coefficients
        coef = svm.coef_.ravel()
        selected_mask = np.abs(coef) > 1e-10
        selected_features = list(np.where(selected_mask)[0])

        solve_time = perf_counter() - start_time

        solution = SVML1Solution(
            selected_features=selected_features,
            n_selected_features=len(selected_features),
            coef=coef,
            svm=svm,
            cv_score=cv_score,
            solve_time=solve_time,
        )

        # The objective is -cv_score (minimized)
        objective = -cv_score

        return solution, objective

    def extract_train_metrics(self, solution: SVML1Solution) -> dict[str, Any]:
        """Extract search metrics from the training result."""
        return cv_score_train_metrics(solution)

    def select_best(self, results_df: pd.DataFrame) -> dict[float, dict[str, Any]]:
        """Select, for each SNR, the C with the smallest mean objective (-CV score)."""
        return select_min_mean_objective(results_df, "C")

    def save(self, solution: SVML1Solution, path: Path) -> None:
        """Save the trained model."""
        import joblib

        meta = {
            **selection_meta(self.name, solution),
            "cv_score": float(solution.cv_score),
            "solve_time": float(solution.solve_time),
        }
        write_model_meta(path, meta)

        # model.npz (coefficients)
        np.savez(path / "model.npz", coef=solution.coef)

        # svm.joblib (trained SVM)
        joblib.dump(solution.svm, path / "svm.joblib")

    def load(self, path: Path) -> SVML1Solution:
        """Load a trained model."""
        import joblib

        meta = read_model_meta(path)

        # model.npz
        data = np.load(path / "model.npz")
        coef = data["coef"]

        # svm.joblib
        svm = joblib.load(path / "svm.joblib")

        return SVML1Solution(
            selected_features=meta["selected_features"],
            n_selected_features=meta["n_selected"],
            coef=coef,
            svm=svm,
            cv_score=meta["cv_score"],
            solve_time=meta["solve_time"],
        )

    def predict(self, solution: SVML1Solution, X: NDArray) -> NDArray:
        """Predict."""
        # Use all features (SVM-L1 holds sparse coefficients internally)
        predictions = solution.svm.predict(X)
        return predictions

    def evaluate(
        self,
        solution: SVML1Solution,
        X_test: NDArray,
        y_test: NDArray,
        true_support: list[int] | None,
        settings: AppSettings,
    ) -> dict[str, Any]:
        """Evaluate."""
        # Predict
        predictions = self.predict(solution, X_test)

        # Decision values (all features)
        decision_values = solution.svm.decision_function(X_test)

        return evaluate_cv_score_solution(
            solution, y_test, predictions, decision_values, true_support
        )
