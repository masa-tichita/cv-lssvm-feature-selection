"""LogReg model spec (LogReg-AIC / LogReg-BIC).

ModelSpec implementation of MIO-based feature selection for logistic regression
(Sato et al., 2016, Comput. Optim. Appl.).

IC(S) = -2 max{ℓ(b, w)} + F(|S| + 1)
- AIC: F = 2 (fixed)
- BIC: F = ln(n) (fixed by the sample size n)

No hyperparameters: F is determined by the choice of information criterion.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from modules.algo.logreg.solver import LogRegMILO
from modules.algo.logreg.types import LogRegParams, LogRegSolution
from modules.models.common import (
    evaluate_mio_solution,
    gurobi_log_dir,
    mio_config_kwargs,
    mio_meta,
    mio_solution_fields,
    mio_train_metrics,
    selection_meta,
)
from modules.experiment.layout import read_model_meta, write_model_meta
from modules.settings import AppSettings


class LogRegSpec:
    """LogReg model spec (LogReg-AIC / LogReg-BIC).

    Paper: Feature subset selection for logistic regression via mixed integer optimization
    (Sato et al., 2016)

    F is a constant determined by the information criterion (AIC/BIC).
    - AIC: F = 2
    - BIC: F = ln(n)
    """

    def __init__(self, name: str, get_penalty_f: Callable[[int], float]) -> None:
        self.name = name
        self._get_penalty_f = get_penalty_f
        self._solver = LogRegMILO()

    def get_param_grid(self, settings: AppSettings) -> dict[str, list[Any]]:
        """Return the search space (no hyperparameters)."""
        _ = settings
        return {}

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        params: dict[str, Any],
        settings: AppSettings,
    ) -> tuple[LogRegSolution, float]:
        """Train the model.

        F is determined by the information criterion given to the constructor.
        A single MILO solve suffices.

        Returns:
            (solution, objective_value)
        """
        config = settings.models.logreg.config
        logreg_params = LogRegParams(
            penalty_F=self._get_penalty_f(X_train.shape[0]),
            tangent_set=config.tangent_set,
            log_dir=gurobi_log_dir(params, self.name),
            **mio_config_kwargs(config),
        )
        solution = self._solver.train(X=X_train, y=y_train, params=logreg_params)
        return solution, solution.obj_value

    def extract_train_metrics(self, solution: LogRegSolution) -> dict[str, Any]:
        """Extract search metrics from the training result."""
        return mio_train_metrics(solution)

    def select_best(self, results_df: pd.DataFrame) -> dict[float, dict[str, Any]]:
        """Select the best parameters for each SNR (no hyperparameters)."""
        return {float(snr): {} for snr in results_df["snr"].unique()}

    def save(self, solution: LogRegSolution, path: Path) -> None:
        """Save the trained model (meta.json; model.npz with z, w, b)."""
        write_model_meta(path, {**selection_meta(self.name, solution), **mio_meta(solution)})
        # The scalar b is stored as an array
        np.savez(path / "model.npz", z=solution.z, w=solution.w, b=np.array([solution.b]))

    def load(self, path: Path) -> LogRegSolution:
        """Load a trained model."""
        data = np.load(path / "model.npz")
        return LogRegSolution(
            z=data["z"],
            w=data["w"],
            b=float(data["b"][0]),
            **mio_solution_fields(read_model_meta(path)),
        )

    def predict(self, solution: LogRegSolution, X: NDArray) -> NDArray:
        """Predict."""
        predictions, _ = self._solver.predict(X, solution)
        return predictions

    def evaluate(
        self,
        solution: LogRegSolution,
        X_test: NDArray,
        y_test: NDArray,
        true_support: list[int] | None,
        settings: AppSettings,
    ) -> dict[str, Any]:
        """Evaluate the logistic regression model of the MIO solution on the test data."""
        predictions, decision_values = self._solver.predict(X_test, solution)
        return evaluate_mio_solution(
            solution, y_test, predictions, decision_values, true_support
        )
