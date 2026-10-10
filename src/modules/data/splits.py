"""K-fold splitting of the training data.

Shared by the proposed method, SVM-CV, and the common L2-SVM refit so that all
of them use the same folds for the same (n, labels, K, shuffle, seed).
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold


def split_k_fold_indices(
    n_samples: int,
    labels: NDArray[np.float64],
    n_folds: int,
    shuffle: bool,
    random_seed: int | None,
) -> tuple[list[NDArray[np.intp]], list[NDArray[np.intp]]]:
    """Split into K folds and return the training and validation index arrays.

    StratifiedKFold is used when there are at least two classes and every class
    has at least ``n_folds`` samples; otherwise plain KFold is used.
    The seed is used only when ``shuffle`` is True.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_samples < n_folds:
        raise ValueError(f"n_samples ({n_samples}) < n_folds ({n_folds})")

    seed = random_seed if shuffle else None
    unique_labels, counts = np.unique(labels, return_counts=True)
    if unique_labels.size >= 2 and np.all(counts >= n_folds):
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        split_iter = splitter.split(np.zeros(n_samples), labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        split_iter = splitter.split(np.zeros(n_samples))

    train_folds: list[NDArray[np.intp]] = []
    val_folds: list[NDArray[np.intp]] = []
    for train_idx, val_idx in split_iter:
        train_folds.append(np.asarray(train_idx, dtype=np.intp))
        val_folds.append(np.asarray(val_idx, dtype=np.intp))
    return train_folds, val_folds
