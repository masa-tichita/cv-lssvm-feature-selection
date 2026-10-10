"""Experiment trials shared by all methods.

* Pre-selection of valid trial seeds with model-independent criteria
  (before any method is trained).
* Success summaries of the finished trials and the trials on which every
  method succeeded.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from modules.data.factory import (
    DataConfig,
    SyntheticDataset,
    build_run_data_config,
    create_dataset,
)
from modules.experiment.layout import ResultLayout
from modules.settings import AppSettings, TrialSelectionConfig

TrialSeedsBySNR = dict[float, list[int]]

_SUMMARY_METRICS = [
    "model_auc",
    "model_f1",
    "feature_f1",
    "n_selected",
    "solve_time",
]


@dataclass(frozen=True)
class TrialValidation:
    """Validity check for one generated dataset before training."""

    accepted: bool
    reason: str
    train_class_counts: dict[float, int]
    test_class_counts: dict[float, int]


def validate_trial_dataset(
    dataset: SyntheticDataset,
    *,
    n_folds: int,
    require_test_two_classes: bool = True,
) -> TrialValidation:
    """Check before training whether fixed stratified CV and AUC evaluation can be applied."""
    train_counts = _class_counts(dataset.y_train)
    test_counts = _class_counts(dataset.y_test)

    arrays = (dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
    if not all(np.isfinite(values).all() for values in arrays):
        return TrialValidation(False, "non_finite_values", train_counts, test_counts)
    if len(train_counts) != 2:
        return TrialValidation(
            False, "train_requires_two_classes", train_counts, test_counts
        )
    if min(train_counts.values()) < n_folds:
        return TrialValidation(
            False,
            f"train_min_class_count_below_{n_folds}",
            train_counts,
            test_counts,
        )
    if require_test_two_classes and len(test_counts) != 2:
        return TrialValidation(
            False, "test_requires_two_classes", train_counts, test_counts
        )
    return TrialValidation(True, "accepted", train_counts, test_counts)


def select_valid_trial_seeds(
    data_config: DataConfig,
    snr_values: Iterable[float],
    config: TrialSelectionConfig,
) -> tuple[TrialSeedsBySNR, pd.DataFrame]:
    """For each SNR, accept the first ``target_trials`` valid trials in generation order."""
    seeds_by_snr: TrialSeedsBySNR = {}
    manifest_rows: list[dict[str, object]] = []

    for raw_snr in snr_values:
        snr = float(raw_snr)
        accepted_seeds: list[int] = []
        stop = config.candidate_seed_start + config.max_candidates_per_snr

        for seed in range(config.candidate_seed_start, stop):
            dataset = create_dataset(
                build_run_data_config(data_config, snr=snr, seed=seed)
            )
            validation = validate_trial_dataset(
                dataset,
                n_folds=config.n_folds,
                require_test_two_classes=config.require_test_two_classes,
            )
            accepted_index: int | None = None
            if validation.accepted:
                accepted_seeds.append(seed)
                accepted_index = len(accepted_seeds)

            manifest_rows.append(
                {
                    "snr": snr,
                    "candidate_seed": seed,
                    "accepted": validation.accepted,
                    "accepted_trial_index": accepted_index,
                    "reason": validation.reason,
                    "train_negative": validation.train_class_counts.get(-1.0, 0),
                    "train_positive": validation.train_class_counts.get(1.0, 0),
                    "test_negative": validation.test_class_counts.get(-1.0, 0),
                    "test_positive": validation.test_class_counts.get(1.0, 0),
                }
            )

            if len(accepted_seeds) == config.target_trials:
                break

        if len(accepted_seeds) != config.target_trials:
            raise RuntimeError(
                f"SNR={snr}: {config.max_candidates_per_snr} candidates yielded "
                f"only {len(accepted_seeds)}/{config.target_trials} valid trials."
            )
        seeds_by_snr[snr] = accepted_seeds

    return seeds_by_snr, pd.DataFrame(manifest_rows)


def prepare_trial_seeds(
    settings: AppSettings,
    result_dir: Path,
) -> TrialSeedsBySNR:
    """Fix the per-SNR seeds shared by all methods before any model is trained.

    Reuses ``<result_dir>/data/trial_selection_manifest.csv`` when it exists
    (resumed runs); otherwise selects the seeds and writes the manifest.
    """
    snr_values = [float(snr) for snr in settings.data.snr_list]
    trial_selection = (
        settings.experiment.trial_selection if settings.experiment else None
    )
    manifest_path = ResultLayout(result_dir).trial_manifest_csv
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        accepted = manifest[manifest["accepted"].astype(bool)]
        seeds_by_snr = {
            snr: [
                int(seed)
                for seed in accepted.loc[
                    accepted["snr"].astype(float) == snr,
                    "candidate_seed",
                ].tolist()
            ]
            for snr in snr_values
        }
        logger.info(f"Reusing trial manifest: {manifest_path}")
        return seeds_by_snr
    if trial_selection is None or not trial_selection.enabled:
        seeds = [int(seed) for seed in settings.data.selection_seeds]
        return {snr: seeds for snr in snr_values}

    seeds_by_snr, manifest = select_valid_trial_seeds(
        settings.data,
        snr_values,
        trial_selection,
    )
    manifest_path.parent.mkdir(exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    for snr, seeds in seeds_by_snr.items():
        rejected = int(((manifest["snr"] == snr) & (~manifest["accepted"])).sum())
        logger.info(
            f"SNR={snr}: selected {len(seeds)} valid trials; "
            f"rejected {rejected} candidates"
        )
    return seeds_by_snr


def build_trial_success_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Return the numbers of successes and failures and the success rate by model and SNR."""
    required = {"model", "snr", "seed"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_results.csv missing columns: {sorted(missing)}")

    status = _status_series(eval_df)
    trials = eval_df[["model", "snr", "seed"]].copy()
    trials["run_status"] = status
    trials = trials.drop_duplicates(["model", "snr", "seed"])

    rows: list[dict[str, object]] = []
    for (model, snr), group in trials.groupby(["model", "snr"], sort=True):
        succeeded = int((group["run_status"] == "success").sum())
        total = len(group)
        rows.append(
            {
                "model": model,
                "snr": snr,
                "trial_n": total,
                "success_n": succeeded,
                "failed_n": total - succeeded,
                "success_rate": succeeded / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def select_common_success_trials(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows of seeds on which all compared models succeeded, for each SNR."""
    if eval_df.empty:
        return eval_df.copy()

    models = set(eval_df["model"].dropna().astype(str).unique())
    status = _status_series(eval_df)
    successful = eval_df[status == "success"].copy()
    common_keys: list[tuple[float, int]] = []
    for (snr, seed), group in successful.groupby(["snr", "seed"], sort=True):
        if set(group["model"].dropna().astype(str)) == models:
            common_keys.append((float(snr), int(seed)))

    if not common_keys:
        return successful.iloc[0:0].copy()
    key_index = pd.MultiIndex.from_tuples(common_keys, names=["snr", "seed"])
    row_index = pd.MultiIndex.from_frame(successful[["snr", "seed"]])
    return successful[row_index.isin(key_index)].copy()


def build_common_success_summary(common_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the main metrics on commonly successful trials (mean, SD, SE)."""
    available_metrics = [
        metric for metric in _SUMMARY_METRICS if metric in common_df.columns
    ]
    columns = ["model", "snr", "common_trial_n"]
    for metric in available_metrics:
        columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_se"])
    if common_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (model, snr), group in common_df.groupby(["model", "snr"], sort=True):
        row: dict[str, object] = {
            "model": model,
            "snr": snr,
            "common_trial_n": group["seed"].nunique(),
        }
        for metric in available_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            std = float(values.std()) if n > 1 else float("nan")
            row[f"{metric}_mean"] = float(values.mean()) if n else float("nan")
            row[f"{metric}_std"] = std
            row[f"{metric}_se"] = std / n**0.5 if n > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _status_series(eval_df: pd.DataFrame) -> pd.Series:
    if "run_status" not in eval_df.columns:
        return pd.Series("success", index=eval_df.index, dtype="object")
    return eval_df["run_status"].fillna("success").astype(str)


def _class_counts(values: np.ndarray) -> dict[float, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {float(label): int(count) for label, count in zip(labels, counts)}
