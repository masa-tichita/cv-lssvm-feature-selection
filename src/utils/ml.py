"""Machine learning utilities.

Helpers shared across models and the common refit.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def normalize_binary_labels(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize labels to ±1 (> 0 → +1, otherwise -1; e.g. {0, 1} → {-1, +1})."""
    return np.where(y > 0, 1.0, -1.0)


def linear_decision(
    X: NDArray[np.float64],
    w: NDArray[np.float64],
    b: float,
) -> NDArray[np.float64]:
    """Decision values X @ w + b of a linear classifier."""
    return X @ w + b


def predict_labels(decision: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predicted labels sign(decision), with a decision value of exactly 0 mapped to +1."""
    predictions = np.sign(decision)
    predictions[predictions == 0] = 1.0
    return predictions


def safe_n_folds(y: NDArray, requested_folds: int) -> int:
    """Choose a safe number of folds given the class sample sizes.

    Stratified K-fold requires at least k samples in each class.
    The number of folds is reduced if necessary, with a warning.

    Args:
        y: Label array.
        requested_folds: Requested number of folds.

    Returns:
        Safe number of folds.

    Raises:
        ValueError: If the smallest class has fewer than 2 samples, so CV is impossible.
    """
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())

    if min_class_count < 2:
        raise ValueError(
            f"Cross-validation requires at least 2 samples per class, "
            f"but minimum class count is {min_class_count}. "
            f"Consider using a larger dataset."
        )

    if min_class_count < requested_folds:
        safe_folds = min_class_count
        logger.warning(
            f"Reducing n_folds from {requested_folds} to {safe_folds} "
            f"(min class count: {min_class_count})"
        )
        return safe_folds
    return requested_folds
