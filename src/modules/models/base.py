"""Model abstraction layer.

Provides the ModelSpec protocol and the model registry.
Adding a new model only requires a model file and one register call.
"""

from itertools import product
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from numpy.typing import NDArray

from modules.settings import AppSettings


class ModelSpec(Protocol):
    """Model spec interface.

    Each model (proposed_hl, svm_cv, svm_l1, etc.) implements this protocol.
    """

    name: str

    # Phase 1: parameter selection
    def get_param_grid(self, settings: AppSettings) -> dict[str, list[Any]]:
        """Return the search space.

        Returns:
            {"gamma": [0.001, 0.01, ...], ...}
        """
        ...

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        params: dict[str, Any],
        settings: AppSettings,
    ) -> tuple[Any, float]:
        """Train the model.

        Args:
            X_train: Training data.
            y_train: Training labels.
            params: Hyperparameters {"gamma": 0.1, ...}.
            settings: Application settings.

        Returns:
            (solution, objective_value): trained model and objective value.
        """
        ...

    def extract_train_metrics(self, solution: Any) -> dict[str, Any]:
        """Extract search metrics from the training result.

        Returns:
            {"n_selected": 5, "solve_time": 1.23, ...}
        """
        ...

    def select_best(self, results_df: pd.DataFrame) -> dict[float, dict[str, Any]]:
        """Select the best parameters for each SNR.

        Args:
            results_df: All search results (columns: snr, seed, gamma, objective, ...).

        Returns:
            {snr: {"gamma": best_gamma, ...}, ...}
        """
        ...

    # Save and load
    def save(self, solution: Any, path: Path) -> None:
        """Save the trained model."""
        ...

    def load(self, path: Path) -> Any:
        """Load a trained model."""
        ...

    # Phase 2: evaluation
    def predict(self, solution: Any, X: NDArray) -> NDArray:
        """Predict."""
        ...

    def evaluate(
        self,
        solution: Any,
        X_test: NDArray,
        y_test: NDArray,
        true_support: list[int] | None,
        settings: AppSettings,
    ) -> dict[str, Any]:
        """Evaluate.

        Args:
            solution: Trained model.
            X_test: Test data.
            y_test: Test labels.
            true_support: True support (for synthetic data).
            settings: Application settings.

        Returns:
            Dict of evaluation results (objective, n_selected, feature_f1, model_f1, ...).
        """
        ...


# Registry: model name → ModelSpec instance
_MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    """Register a model in the registry."""
    _MODEL_REGISTRY[spec.name] = spec


def get_model(name: str) -> ModelSpec:
    """Get a model from the registry.

    Raises:
        KeyError: If the model is not registered.
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """Return the names of registered models."""
    return list(_MODEL_REGISTRY.keys())


def expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into its Cartesian product.

    Args:
        param_grid: {"gamma": [0.1, 1.0], "C": [1, 10]}

    Returns:
        [{"gamma": 0.1, "C": 1}, {"gamma": 0.1, "C": 10}, ...]

    Raises:
        ValueError: If a parameter list is empty.
    """
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Check for empty parameter lists
    for key, val_list in zip(keys, values):
        if not val_list:
            raise ValueError(
                f"Parameter '{key}' has an empty list. "
                f"Please provide at least one value in config.yml."
            )

    return [dict(zip(keys, combo)) for combo in product(*values)]
