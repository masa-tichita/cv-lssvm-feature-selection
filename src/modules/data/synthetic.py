"""Synthetic data generation.

Adapted from the simulation design of Purcaru et al. (2026),
J. Optim. Theory Appl. 210:16: labels are the sign of a noisy linear score
(class balance is not enforced). In our adaptation the uniform components are
scaled by sqrt(3) to unit variance, the covariance is given by
Sigma_jl = rho^|j-l|, and the noise variance sigma^2 is set from the SNR.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

SupportMode = Literal["spaced", "random"]
UniformScaling = Literal["purcaru", "unit_variance"]


class SyntheticDataset(BaseModel):
    """Generated synthetic data."""

    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]
    true_support: list[int]
    coefficients: NDArray[np.float64]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SyntheticConfig(BaseModel):
    """Synthetic data setting.

    Adapted from the simulation design of Purcaru et al. (2026),
    J. Optim. Theory Appl. 210:16.
    Generates X = U L^T with Sigma = L L^T, Sigma_jl = rho^|j-l|,
    and y = sign(Xw + b + eps) with Var(eps) = w^T Sigma w / SNR.

    ``uniform_scaling="purcaru"`` uses U_j ~ Uniform(-1, 1) as stated in that paper;
    ``"unit_variance"`` (used in our experiments) uses
    U_j ~ Uniform(-sqrt(3), sqrt(3)) so that Cov(X) = Sigma.
    """

    generator: Literal["synthetic"] = "synthetic"
    train_size: int = Field(default=100, ge=1)
    test_size: int = Field(default=1000, ge=1)
    features: int = Field(default=20, ge=1)
    informative: int = Field(default=10, ge=1)
    # Correlation: Sigma_jl = rho^|j-l|
    rho: float = Field(default=0.5, ge=0.0, lt=1.0)
    support_mode: SupportMode = "spaced"
    support_seed: int | None = None
    uniform_scaling: UniformScaling = "purcaru"
    snr: float = Field(default=2.0, gt=0.0)
    seed: int | None = None

    # Lists for experiments
    snr_list: list[float] = Field(default_factory=list)
    selection_seeds: list[int] = Field(default_factory=list)

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def check_allocations(self) -> "SyntheticConfig":
        if self.features < self.informative:
            raise ValueError("features must be at least informative")
        return self

    def select_support(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Return the support set."""
        if self.support_mode == "random":
            local_rng = (
                np.random.default_rng(self.support_seed)
                if self.support_seed is not None
                else rng
            )
            support = local_rng.choice(
                np.arange(self.features, dtype=np.int64),
                size=self.informative,
                replace=False,
            )
            support.sort()
            return support.astype(np.int64)

        if self.informative == 1:
            return np.array([0], dtype=np.int64)
        return np.rint(np.linspace(0, self.features - 1, self.informative)).astype(
            np.int64
        )


def _generate_covariance_matrix(config: SyntheticConfig) -> NDArray[np.float64]:
    """Covariance matrix Sigma_jl = rho^|j-l|."""
    indices = np.arange(config.features, dtype=np.int64)
    distances = np.abs(indices[:, None] - indices[None, :])
    return (config.rho**distances).astype(np.float64)


def create_synthetic_dataset(config: SyntheticConfig) -> SyntheticDataset:
    """Generate one synthetic dataset (training and test samples)."""
    rng = np.random.default_rng(config.seed)
    covariance = _generate_covariance_matrix(config)
    chol = np.linalg.cholesky(covariance)
    size = config.train_size + config.test_size

    support = config.select_support(rng)
    coefficients = np.zeros(config.features, dtype=np.float64)
    coefficients[support] = rng.uniform(-1.0, 1.0, size=config.informative)
    coef_norm = float(np.linalg.norm(coefficients))
    if coef_norm == 0.0:
        coefficients[support[0]] = 1.0
    else:
        coefficients /= coef_norm

    intercept = float(rng.uniform(-1.0, 1.0))
    uniform_half_width = (
        float(np.sqrt(3.0)) if config.uniform_scaling == "unit_variance" else 1.0
    )
    features = (
        rng.uniform(
            -uniform_half_width,
            uniform_half_width,
            size=(size, config.features),
        )
        @ chol.T
    )

    signal_variance = float(coefficients @ covariance @ coefficients)
    noise_std = float(np.sqrt(signal_variance / config.snr))
    noise = rng.normal(0.0, noise_std, size=size)
    margins = features @ coefficients + intercept + noise
    labels = np.where(margins >= 0.0, 1.0, -1.0)

    if np.all(labels == 1.0) or np.all(labels == -1.0):
        flip_index = int(np.abs(margins).argmin())
        labels[flip_index] = -labels[flip_index]

    permutation = rng.permutation(size)
    features = features[permutation]
    labels = labels[permutation]

    return SyntheticDataset(
        X_train=features[: config.train_size],
        X_test=features[config.train_size :],
        y_train=labels[: config.train_size],
        y_test=labels[config.train_size :],
        true_support=support.tolist(),
        coefficients=coefficients,
    )


def build_synthetic_config(
    base: SyntheticConfig,
    snr: float,
    seed: int,
) -> SyntheticConfig:
    """Build a data config (with snr and seed overridden)."""
    return base.model_copy(update={"snr": snr, "seed": seed})
