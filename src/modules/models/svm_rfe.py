"""SVM-RFECV model spec (SVM-RFE).

ModelSpec implementation of recursive feature elimination with cross-validation
using a linear SVM. The number of features is chosen automatically from the CV score.
"""

import signal
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

from modules.models.common import (
    cv_score_train_metrics,
    evaluate_cv_score_solution,
    select_min_mean_objective,
    selection_meta,
)
from modules.experiment.layout import read_model_meta, write_model_meta
from modules.settings import AppSettings
from utils.ml import safe_n_folds


class _TrainTimeout(Exception):
    """Training timeout of svm_rfe."""


@dataclass
class SVMRFECVSolution:
    """Training result of SVM-RFECV."""

    # Indices of the selected features
    selected_features: list[int]
    n_selected_features: int
    # Feature ranking (1 is the most important)
    ranking: NDArray[np.int64]
    # Trained SVM (trained on the selected features)
    svm: SVC
    # CV score at the optimal number of features
    cv_score: float
    # CV score for each number of features (for analysis)
    cv_scores_per_n_features: NDArray[np.float64]
    # Training time
    solve_time: float


class SVMRFESpec:
    """SVM-RFECV model spec."""

    name = "svm_rfe"

    def get_param_grid(self, settings: AppSettings) -> dict[str, list[Any]]:
        """Return the search space."""
        s = settings.models.svm_rfe
        return {"C": s.search.C_list}

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        params: dict[str, Any],
        settings: AppSettings,
    ) -> tuple[SVMRFECVSolution, float]:
        """Train the model.

        Use RFECV to choose the number of features automatically:
        the number with the largest internal CV score is selected.

        Returns:
            (solution, objective_value): objective is -cv_score (minimized).
        """
        s = settings.models.svm_rfe
        config = s.config

        C = params["C"]
        time_limit = s.time_limit

        start_time = perf_counter()

        # Create the SVM
        svm = SVC(kernel="linear", C=C, random_state=config.cv_seed)

        # Feature selection with RFECV (number of features chosen by the CV score)
        # Adjust the number of folds to the class sample sizes
        n_folds = safe_n_folds(y_train, config.n_folds)
        rfecv = RFECV(
            estimator=svm,
            step=s.step,
            cv=n_folds,
            scoring="accuracy",
            min_features_to_select=1,
        )

        # Run with a timeout
        def _timeout_handler(signum: int, frame: object) -> None:
            raise _TrainTimeout(f"svm_rfe training exceeded {time_limit}s")

        timed_out = False
        old_handler = None
        if time_limit is not None:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, time_limit)

        try:
            rfecv.fit(X_train, y_train)
        except _TrainTimeout:
            timed_out = True
            logger.warning(f"svm_rfe training timed out after {time_limit}s (C={C})")
        finally:
            if time_limit is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel the timer
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        if timed_out:
            solve_time = perf_counter() - start_time
            # On timeout, return a dummy solution with all features and score 0
            p = X_train.shape[1]
            all_features = list(range(p))
            svm_fallback = SVC(kernel="linear", C=C, random_state=config.cv_seed)
            svm_fallback.fit(X_train, y_train)
            solution = SVMRFECVSolution(
                selected_features=all_features,
                n_selected_features=p,
                ranking=np.ones(p, dtype=np.int64),
                svm=svm_fallback,
                cv_score=0.0,
                cv_scores_per_n_features=np.zeros(p, dtype=np.float64),
                solve_time=solve_time,
            )
            return solution, 0.0

        # Selected features
        selected_mask = rfecv.support_
        selected_features = list(np.where(selected_mask)[0])

        # CV score at the optimal number of features
        cv_score = float(rfecv.cv_results_["mean_test_score"].max())

        # Train the final SVM on the selected features
        X_selected = X_train[:, selected_mask]
        svm_final = SVC(kernel="linear", C=C, random_state=config.cv_seed)
        svm_final.fit(X_selected, y_train)

        solve_time = perf_counter() - start_time

        solution = SVMRFECVSolution(
            selected_features=selected_features,
            n_selected_features=len(selected_features),
            ranking=rfecv.ranking_,
            svm=svm_final,
            cv_score=cv_score,
            cv_scores_per_n_features=rfecv.cv_results_["mean_test_score"],
            solve_time=solve_time,
        )

        # The objective is -cv_score (minimized)
        objective = -cv_score

        return solution, objective

    def extract_train_metrics(self, solution: SVMRFECVSolution) -> dict[str, Any]:
        """Extract search metrics from the training result."""
        return cv_score_train_metrics(solution)

    def select_best(self, results_df: pd.DataFrame) -> dict[float, dict[str, Any]]:
        """Select, for each SNR, the C with the smallest mean objective (-CV score)."""
        return select_min_mean_objective(results_df, "C")

    def save(self, solution: SVMRFECVSolution, path: Path) -> None:
        """Save the trained model."""
        import joblib

        meta = {
            **selection_meta(self.name, solution),
            "cv_score": float(solution.cv_score),
            "solve_time": float(solution.solve_time),
        }
        write_model_meta(path, meta)

        # model.npz (ranking + CV score for each number of features)
        np.savez(
            path / "model.npz",
            ranking=solution.ranking,
            cv_scores_per_n_features=solution.cv_scores_per_n_features,
        )

        # svm.joblib (trained SVM)
        joblib.dump(solution.svm, path / "svm.joblib")

    def load(self, path: Path) -> SVMRFECVSolution:
        """Load a trained model."""
        import joblib

        meta = read_model_meta(path)

        # model.npz
        data = np.load(path / "model.npz")
        ranking = data["ranking"]
        cv_scores_per_n_features = data["cv_scores_per_n_features"]

        # svm.joblib
        svm = joblib.load(path / "svm.joblib")

        return SVMRFECVSolution(
            selected_features=meta["selected_features"],
            n_selected_features=meta["n_selected"],
            ranking=ranking,
            svm=svm,
            cv_score=meta["cv_score"],
            cv_scores_per_n_features=cv_scores_per_n_features,
            solve_time=meta["solve_time"],
        )

    def predict(self, solution: SVMRFECVSolution, X: NDArray) -> NDArray:
        """Predict."""
        # Use only the selected features
        X_selected = X[:, solution.selected_features]
        predictions = solution.svm.predict(X_selected)
        return predictions

    def evaluate(
        self,
        solution: SVMRFECVSolution,
        X_test: NDArray,
        y_test: NDArray,
        true_support: list[int] | None,
        settings: AppSettings,
    ) -> dict[str, Any]:
        """Evaluate."""
        # Predict
        predictions = self.predict(solution, X_test)

        # Decision values (selected features only)
        X_selected = X_test[:, solution.selected_features]
        decision_values = solution.svm.decision_function(X_selected)

        return evaluate_cv_score_solution(
            solution, y_test, predictions, decision_values, true_support
        )
