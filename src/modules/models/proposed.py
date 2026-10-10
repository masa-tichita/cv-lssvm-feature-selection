"""Model spec of the proposed method.

ModelSpec implementation of Proposed-HL (``proposed_hl``; hinge-loss-based
validation model, MILO) and Proposed-LS (``proposed_ls``; least-squares-based
validation model, MIQO). Both share the settings in ``models.proposed``.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from numpy.typing import NDArray

from modules.algo.proposed.solver import ProposedSolver
from modules.algo.proposed.types import ProposedParams, ProposedSolution, Variant
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


class ProposedSpec:
    """Model spec of Proposed-HL / Proposed-LS."""

    def __init__(self, name: str, variant: Variant) -> None:
        self.name = name
        self.variant: Variant = variant
        self._solver = ProposedSolver()

    def get_param_grid(self, settings: AppSettings) -> dict[str, list[Any]]:
        """Return the search space."""
        return {"gamma": settings.models.proposed.search.gamma_list}

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        params: dict[str, Any],
        settings: AppSettings,
    ) -> tuple[ProposedSolution, float]:
        """Train the model.

        Returns:
            (solution, objective_value)
        """
        config = settings.models.proposed.config
        full_params = ProposedParams(
            gamma=params["gamma"],
            variant=self.variant,
            n_folds=config.n_folds,
            shuffle=config.shuffle,
            seed=config.cv_seed,
            var_bound_w=config.var_bound_w,
            var_bound_b=config.var_bound_b,
            bigm_link=config.bigm_link,
            log_dir=gurobi_log_dir(params, self.name),
            **mio_config_kwargs(config),
        )
        solution = self._solver.train(X=X_train, y=y_train, params=full_params)
        return solution, solution.obj_value

    def extract_train_metrics(self, solution: ProposedSolution) -> dict[str, Any]:
        """Extract search metrics from the training result."""
        return mio_train_metrics(solution)

    def select_best(self, results_df: pd.DataFrame) -> dict[float, dict[str, Any]]:
        """Select, for each SNR, the gamma with the smallest mean objective."""
        return select_min_mean_objective(results_df, "gamma")

    def save(self, solution: ProposedSolution, path: Path) -> None:
        """Save the trained model (meta.json; model.npz with z and the fold models)."""
        write_model_meta(path, {**selection_meta(self.name, solution), **mio_meta(solution)})
        save_fold_models(path, solution.z, solution.fold_models)

    def load(self, path: Path) -> ProposedSolution:
        """Load a trained model."""
        z, fold_models = load_fold_models(path)
        return ProposedSolution(
            z=z, fold_models=fold_models, **mio_solution_fields(read_model_meta(path))
        )

    def predict(self, solution: ProposedSolution, X: NDArray) -> NDArray:
        """Predict with the mean decision value of the per-fold models."""
        predictions, _ = self._solver.predict(X, solution)
        return predictions

    def evaluate(
        self,
        solution: ProposedSolution,
        X_test: NDArray,
        y_test: NDArray,
        true_support: list[int] | None,
        settings: AppSettings,
    ) -> dict[str, Any]:
        """Evaluate the per-fold models of the MIO solution on the test data."""
        predictions, decision_values = self._solver.predict(X_test, solution)
        return evaluate_mio_solution(
            solution, y_test, predictions, decision_values, true_support
        )
