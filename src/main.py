"""Command-line interface.

Examples (poe tasks defined in pyproject.toml wrap these commands):
    uv run poe run --config configs/p10_theta5_rho_0.35.yml
    uv run poe refit results/<result_dir>
    uv run poe figures results/<result_dir>
    uv run poe table results/<dir1> results/<dir2> ... --output_dir paper_outputs
"""

import fire

from apps.experiment import run_experiment
from apps.figures import build_figures
from apps.refit import run_common_refit
from apps.table import build_timing_table
from modules.settings import set_config_path
from utils.logging import setup_logger


class Commands:
    """Experiments for cross-validated feature selection with LS-SVMs."""

    def run(
        self,
        config: str,
        model: str | None = None,
        output_dir: str = "results",
        resume_from: str | None = None,
    ) -> None:
        """Run all methods of one experimental condition.

        Args:
            config: Path to a YAML config (e.g. configs/p10_theta5_rho_0.35.yml).
            model: Run only this method (proposed_hl, proposed_ls, svm_cv, svm_l1,
                svm_rfe, logreg_aic, logreg_bic).
            output_dir: Parent directory of the result directory.
            resume_from: Result directory of an interrupted run to resume.
        """
        set_config_path(config)
        result_path = run_experiment(
            model_name=model,
            output_dir=output_dir,
            resume_from=resume_from,
        )
        print(f"Results saved to: {result_path}")

    def refit(self, result_dir: str, output_dir: str | None = None) -> None:
        """Common L2-SVM refit on the saved feature subsets (test AUC).

        Args:
            result_dir: Result directory created by ``run``.
            output_dir: Output directory (default: <result_dir>/refit).
        """
        out = run_common_refit(result_dir, output_dir)
        print(f"Refit results saved to: {out}")

    def figures(self, result_dir: str, output_dir: str | None = None) -> None:
        """AUC / F1 / subset-size panels for each SNR of one result directory.

        Args:
            result_dir: Result directory (``refit`` must have been run).
            output_dir: Output directory (default: <result_dir>/figures).
        """
        out = build_figures(result_dir, output_dir)
        print(f"Figures saved to: {out}")

    def table(self, *result_dirs: str, output_dir: str = "paper_outputs") -> None:
        """Computation-time table (Table 1) from several result directories.

        Args:
            result_dirs: Result directories of the experimental conditions.
            output_dir: Output directory for timing_table.csv / timing_table.tex.
        """
        out = build_timing_table(list(result_dirs), output_dir)
        print(f"Timing table saved to: {out}")


def main() -> None:
    setup_logger()
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
