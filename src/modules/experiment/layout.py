"""File layout of an experiment result directory.

Every command that writes or reads a result directory (``run``, ``refit``,
``figures``, ``table``) takes its paths from here::

    <result_dir>/
      config.json, config.yml                 settings snapshot
      data/trial_selection_manifest.csv       pre-selected trial seeds
      training/search_results.csv, best_params.csv
      training/checkpoints/<model>/           resumable progress
      training/gurobi_logs/<model>/snr=<snr>/seed=<seed>/
      training/models/<model>/snr_<snr>/seed_<seed>/   saved model (meta.json, ...)
      evaluation/eval_results.csv, common_success_eval_results.csv
      tables/trial_success_summary.csv, common_success_summary.csv,
             timing_summary.csv, timing_table.csv
      refit/classification_results.csv, classification_summary.csv, protocol.json
      figures/
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultLayout:
    """Paths inside one result directory."""

    root: Path

    # Settings snapshot
    @property
    def config_json(self) -> Path:
        return self.root / "config.json"

    @property
    def config_yml(self) -> Path:
        return self.root / "config.yml"

    def read_config(self) -> dict[str, Any]:
        """Settings snapshot (config.json) as a dict."""
        return json.loads(self.config_json.read_text(encoding="utf-8"))

    # Trial selection
    @property
    def trial_manifest_csv(self) -> Path:
        return self.root / "data" / "trial_selection_manifest.csv"

    # Training
    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def search_results_csv(self) -> Path:
        return self.training_dir / "search_results.csv"

    @property
    def best_params_csv(self) -> Path:
        return self.training_dir / "best_params.csv"

    @property
    def checkpoints_dir(self) -> Path:
        return self.training_dir / "checkpoints"

    def gurobi_log_dir(self, model: str, snr: float, seed: int) -> Path:
        return self.training_dir / "gurobi_logs" / model / f"snr={snr}" / f"seed={seed}"

    @property
    def models_dir(self) -> Path:
        return self.training_dir / "models"

    def model_dir(self, model: str, snr: float, seed: int) -> Path:
        """Directory of the saved model evaluated on one trial."""
        return self.models_dir / model / f"snr_{snr}" / f"seed_{seed}"

    def parse_model_dir(self, meta_path: Path) -> tuple[str, float, int]:
        """(model, snr, seed) of a file inside a directory given by ``model_dir``."""
        parts = meta_path.relative_to(self.models_dir).parts
        return parts[0], float(parts[-3][4:]), int(parts[-2][5:])

    # Evaluation
    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def eval_results_csv(self) -> Path:
        return self.evaluation_dir / "eval_results.csv"

    @property
    def common_success_eval_csv(self) -> Path:
        return self.evaluation_dir / "common_success_eval_results.csv"

    # Tables
    @property
    def tables_dir(self) -> Path:
        return self.root / "tables"

    @property
    def trial_success_summary_csv(self) -> Path:
        return self.tables_dir / "trial_success_summary.csv"

    @property
    def common_success_summary_csv(self) -> Path:
        return self.tables_dir / "common_success_summary.csv"

    @property
    def timing_summary_csv(self) -> Path:
        return self.tables_dir / "timing_summary.csv"

    @property
    def timing_table_csv(self) -> Path:
        return self.tables_dir / "timing_table.csv"

    # Common L2-SVM refit
    @property
    def refit_dir(self) -> Path:
        return self.root / "refit"

    # Figures
    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"


def write_model_meta(model_dir: Path, meta: dict[str, Any]) -> None:
    """Create a saved-model directory and write its meta.json."""
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def read_model_meta(model_dir: Path) -> dict[str, Any]:
    """Read the meta.json of a saved-model directory."""
    with open(model_dir / "meta.json") as f:
        return json.load(f)


# Files written by ``refit`` (relative to its output directory, default ``refit_dir``)
REFIT_RESULTS_CSV = "classification_results.csv"
REFIT_SUMMARY_CSV = "classification_summary.csv"
REFIT_PROTOCOL_JSON = "protocol.json"
