from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from modules.data.factory import DataConfig
from modules.data.synthetic import SyntheticConfig
from utils.system import project_root


class ProposedConfig(BaseModel):
    """Static settings of the proposed method (shared by Proposed-HL and Proposed-LS)."""

    max_features: int | None = Field(default=None, ge=1)
    time_limit: float | None = Field(default=None, ge=0.0)
    opt_gap: float = Field(default=1e-4, ge=0.0)
    n_folds: int = Field(default=5, ge=2)
    shuffle: bool = Field(default=True)
    cv_seed: int = Field(default=42)
    # Bounds on continuous variables (for tightening Big-M)
    var_bound_w: float | None = Field(default=None, gt=0.0)
    var_bound_b: float | None = Field(default=None, gt=0.0)
    # Add Big-M linking constraints (-W*z <= w <= W*z) to tighten the LP relaxation
    bigm_link: bool = Field(default=False)
    threads: int | None = Field(default=None, ge=1)
    nodefile_start: float | None = Field(default=None, ge=0.0)
    soft_mem_limit: float | None = Field(default=None, gt=0.0)
    nodefile_dir: str | None = None

    model_config = ConfigDict(frozen=True)


class TrialSelectionConfig(BaseModel):
    """Settings for selecting trials usable by all methods before training."""

    enabled: bool = False
    target_trials: int = Field(default=100, ge=1)
    n_folds: int = Field(default=5, ge=2)
    candidate_seed_start: int = Field(default=1, ge=0)
    max_candidates_per_snr: int = Field(default=10_000, ge=1)
    require_test_two_classes: bool = True


class ExperimentConfig(BaseModel):
    """Experiment settings.

    experiment:
      name: sample
      models:
        - proposed_hl
        - proposed_ls
    """

    name: str = Field(default="experiment")
    models: list[str] = Field(default_factory=list)
    parameter_selection_scope: Literal["per_seed", "per_snr"] = "per_seed"
    trial_selection: TrialSelectionConfig = Field(default_factory=TrialSelectionConfig)


class HyperParamSpace(BaseModel):
    """Hyperparameter search space."""

    gamma_list: list[float] = Field(default_factory=list)
    C_list: list[float] = Field(default_factory=list)


class ProposedSettings(BaseModel):
    """Settings of the proposed method (static settings + search space of gamma).

    Shared by the two methods proposed_hl (Proposed-HL) and proposed_ls (Proposed-LS).
    """

    config: ProposedConfig = Field(default_factory=ProposedConfig)
    search: HyperParamSpace = Field(default_factory=HyperParamSpace)


class SVMConfig(BaseModel):
    """Common SVM settings."""

    n_folds: int = Field(default=5, ge=2)
    cv_seed: int = Field(default=42)

    model_config = ConfigDict(frozen=True)


class SVMRFESettings(BaseModel):
    """SVM-RFECV settings (number of features chosen by CV)."""

    config: SVMConfig = Field(default_factory=SVMConfig)
    search: HyperParamSpace = Field(default_factory=HyperParamSpace)
    # RFE step (number of features removed per iteration)
    step: int = Field(default=1, ge=1)
    # Training time limit in seconds; exceeding it is treated as a timeout
    time_limit: float | None = Field(default=None, ge=0.0)


class SVML1Settings(BaseModel):
    """SVM-L1 settings."""

    config: SVMConfig = Field(default_factory=SVMConfig)
    search: HyperParamSpace = Field(default_factory=HyperParamSpace)


class LogRegConfig(BaseModel):
    """Static settings of the MIO-based logistic regression (LogReg-AIC / LogReg-BIC)."""

    max_features: int | None = Field(default=None, ge=1)
    time_limit: float | None = Field(default=None, ge=0.0)
    opt_gap: float = Field(default=1e-4, ge=0.0)
    # Tangent point set: "V1" (5 points), "V2" (9 points), "V3" (17 points)
    tangent_set: str = Field(default="V3")
    threads: int | None = Field(default=None, ge=1)
    nodefile_start: float | None = Field(default=None, ge=0.0)
    soft_mem_limit: float | None = Field(default=None, gt=0.0)
    nodefile_dir: str | None = None

    model_config = ConfigDict(frozen=True)


class SVMCVConfig(BaseModel):
    """Static settings of SVM-CV (Purcaru et al., 2026)."""

    max_features: int | None = Field(default=None, ge=1)
    time_limit: float | None = Field(default=500.0, ge=0.0)
    opt_gap: float = Field(default=1e-4, ge=0.0)
    n_folds: int = Field(default=10, ge=2)
    shuffle: bool = Field(default=True)
    cv_seed: int = Field(default=42)
    optimality_tol: float | None = Field(default=1e-7, gt=0.0)
    threads: int | None = Field(default=None, ge=1)
    nodefile_start: float | None = Field(default=None, ge=0.0)
    soft_mem_limit: float | None = Field(default=None, gt=0.0)
    nodefile_dir: str | None = None

    model_config = ConfigDict(frozen=True)


class LogRegSettings(BaseModel):
    """LogReg settings (shared by logreg_aic and logreg_bic).

    No hyperparameters: F is determined by the choice of information criterion.
    """

    config: LogRegConfig = Field(default_factory=LogRegConfig)


class SVMCVSettings(BaseModel):
    """SVM-CV settings."""

    config: SVMCVConfig = Field(default_factory=SVMCVConfig)
    search: HyperParamSpace = Field(default_factory=HyperParamSpace)


class RefitSVMConfig(BaseModel):
    """Settings of the common L2-SVM refit on saved feature subsets."""

    n_folds: int = Field(default=10, ge=2)
    shuffle: bool = Field(default=True)
    cv_seed: int = Field(default=42)
    time_limit: float | None = Field(default=None, ge=0.0)
    opt_gap: float = Field(default=1e-4, ge=0.0)

    model_config = ConfigDict(frozen=True)


class RefitSVMSettings(BaseModel):
    """Common L2-SVM refit settings."""

    config: RefitSVMConfig = Field(default_factory=RefitSVMConfig)
    search: HyperParamSpace = Field(default_factory=HyperParamSpace)


class ModelsConfig(BaseModel):
    """Model settings."""

    proposed: ProposedSettings = Field(default_factory=ProposedSettings)
    svm_cv: SVMCVSettings = Field(default_factory=SVMCVSettings)
    svm_l1: SVML1Settings = Field(default_factory=SVML1Settings)
    svm_rfe: SVMRFESettings = Field(default_factory=SVMRFESettings)
    logreg: LogRegSettings = Field(default_factory=LogRegSettings)
    refit_svm: RefitSVMSettings = Field(default_factory=RefitSVMSettings)


class AppSettings(BaseSettings):
    """Application settings (loaded from config.yml)."""

    model_config = SettingsConfigDict(
        yaml_file=str(project_root() / "config.yml"),
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    data: DataConfig = Field(default_factory=SyntheticConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    experiment: ExperimentConfig | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
        )


_config_path: str | None = None
_settings_cache: AppSettings | None = None


def set_config_path(path: str) -> None:
    """Set the config file path (and clear the cache)."""
    global _config_path, _settings_cache  # noqa: PLW0603
    _config_path = path
    _settings_cache = None


def clear_settings_cache() -> None:
    """Reset the settings cache and the config path."""
    global _config_path, _settings_cache  # noqa: PLW0603
    _config_path = None
    _settings_cache = None


def get_settings() -> AppSettings:
    """Return the settings (cached)."""
    global _settings_cache  # noqa: PLW0603
    if _settings_cache is not None:
        return _settings_cache

    if _config_path is not None:
        import yaml

        with open(_config_path) as f:
            data = yaml.safe_load(f)
        _settings_cache = AppSettings(**(data or {}))
    else:
        _settings_cache = AppSettings()
    return _settings_cache


def get_experiment_config(settings: AppSettings | None = None) -> ExperimentConfig:
    """Return the experiment section of the given (default: loaded) settings; raise if missing."""
    if settings is None:
        settings = get_settings()
    if settings.experiment is None:
        raise ValueError("experiment section not found in config.yml")
    return settings.experiment
