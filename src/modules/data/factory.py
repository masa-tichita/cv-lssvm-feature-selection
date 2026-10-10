"""Common dispatch for experiment data generators."""

from typing import TypeAlias

from modules.data.synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    build_synthetic_config,
    create_synthetic_dataset,
)

DataConfig: TypeAlias = SyntheticConfig

__all__ = [
    "DataConfig",
    "SyntheticDataset",
    "build_result_dir_name",
    "build_run_data_config",
    "create_dataset",
    "get_train_size",
]


def build_run_data_config(
    base: SyntheticConfig,
    snr: float,
    seed: int,
) -> SyntheticConfig:
    """Build a data config with snr and seed overridden for the experiment loop."""
    return build_synthetic_config(base, snr=snr, seed=seed)


def create_dataset(config: SyntheticConfig) -> SyntheticDataset:
    """Generate the dataset specified by the data config."""
    return create_synthetic_dataset(config)


def get_train_size(config: SyntheticConfig) -> int:
    """Return the training size from the data config."""
    return config.train_size


def build_result_dir_name(
    exp_name: str,
    timestamp: str,
    config: SyntheticConfig,
) -> str:
    """Build the result directory name from the data config.

    Example: ``<name>_<timestamp>_n100_p10_theta5_rho_0.35_unitvar``.
    """
    n_train = get_train_size(config)
    base = (
        f"{exp_name}_{timestamp}_n{n_train}"
        f"_p{config.features}_theta{config.informative}"
        f"_rho_{config.rho:.2f}"
    )
    if config.uniform_scaling == "unit_variance":
        base += "_unitvar"
    if config.support_mode == "random" and config.support_seed is not None:
        base += f"_ss_{config.support_seed}"
    return base
