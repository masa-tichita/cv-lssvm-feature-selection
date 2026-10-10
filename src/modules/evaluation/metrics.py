"""Evaluation metrics shared by all methods.

- Feature selection metrics: precision, recall and F1 of the selected features
  against the true support (synthetic data).
- Classification metrics: test accuracy, precision, recall and F1 of each
  method's own predictions, and the AUC of its decision values.

Each method computes its own predictions and decision values (including its
rule for a decision value of exactly 0); these functions only score them.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.ml import normalize_binary_labels


def compute_feature_selection_metrics(
    selected_features: list[int],
    true_support: list[int] | None,
) -> dict[str, Any]:
    """Compute feature selection metrics.

    TP = |selected ∩ true support|, FP = |selected - true support|,
    FN = |true support - selected|. A metric with a zero denominator is 0.

    Args:
        selected_features: Indices of the selected features.
        true_support: True support (indices of the true features).

    Returns:
        Dict of metrics (feature_precision, feature_recall, feature_f1);
        an empty dict if true_support is None.
    """
    if true_support is None:
        return {}

    selected_set = set(selected_features)
    true_set = set(true_support)

    tp = len(selected_set & true_set)
    fp = len(selected_set - true_set)
    fn = len(true_set - selected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "feature_precision": precision,
        "feature_recall": recall,
        "feature_f1": f1,
    }


def compute_classification_metrics(
    y_true: NDArray,
    predictions: NDArray,
    decision_values: NDArray,
) -> dict[str, Any]:
    """Compute test classification metrics of a method's own classifier.

    Labels are normalized to ±1 (> 0 → +1) and +1 is the positive class.
    Precision, recall and F1 are 0 when their denominator is 0. The AUC uses
    the decision values and is None when it is undefined (e.g. a single class
    in ``y_true``).

    Args:
        y_true: True test labels.
        predictions: Predicted labels (±1).
        decision_values: Decision values (larger means more likely +1).

    Returns:
        Dict of metrics (model_accuracy, model_precision, model_recall,
        model_f1, model_auc).
    """
    y_signed = normalize_binary_labels(y_true)
    # ignore: the type stubs for zero_division are outdated
    result: dict[str, Any] = {
        "model_accuracy": float(accuracy_score(y_signed, predictions)),
        "model_precision": float(
            precision_score(y_signed, predictions, zero_division=0.0)  # type: ignore[arg-type]
        ),
        "model_recall": float(
            recall_score(y_signed, predictions, zero_division=0.0)  # type: ignore[arg-type]
        ),
        "model_f1": float(
            f1_score(y_signed, predictions, zero_division=0.0)  # type: ignore[arg-type]
        ),
    }

    y_binary = np.where(y_true > 0, 1, 0)
    try:
        result["model_auc"] = float(roc_auc_score(y_binary, decision_values))
    except ValueError:
        result["model_auc"] = None

    return result


def compute_evaluation_metrics(
    y_true: NDArray,
    predictions: NDArray,
    decision_values: NDArray,
    selected_features: list[int],
    true_support: list[int] | None,
) -> dict[str, Any]:
    """Classification metrics followed by feature selection metrics (one eval row)."""
    return {
        **compute_classification_metrics(y_true, predictions, decision_values),
        **compute_feature_selection_metrics(selected_features, true_support),
    }
