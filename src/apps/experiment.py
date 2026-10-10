"""``run`` command: run all methods of one experimental condition.

Parameter selection and evaluation run in the same phase for each method
(``modules.experiment.runner``); progress is checkpointed so that an
interrupted run can be resumed with ``--resume_from``.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from modules.data.factory import build_result_dir_name
from modules.evaluation.timing import build_timing_display_table, build_timing_summary
from modules.experiment.layout import ResultLayout
from modules.experiment.runner import (
    best_params_rows,
    parameter_selection_scope,
    run_model,
)
from modules.experiment.trials import (
    build_common_success_summary,
    build_trial_success_summary,
    prepare_trial_seeds,
    select_common_success_trials,
)
from modules.models.base import get_model
from modules.models.bootstrap import register_all
from modules.settings import AppSettings, get_experiment_config, get_settings


def _save_settings_snapshot(settings: AppSettings, layout: ResultLayout) -> None:
    """Save the run settings to the result directory."""
    settings_dict = settings.model_dump(mode="json")
    layout.config_json.write_text(
        settings.model_dump_json(indent=2),
        encoding="utf-8",
    )
    layout.config_yml.write_text(
        yaml.safe_dump(
            settings_dict,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def run_experiment(
    settings: AppSettings | None = None,
    model_name: str | None = None,
    experiment_name: str | None = None,
    output_dir: str = "result",
    resume_from: str | None = None,
) -> Path:
    """Run an experiment.

    Args:
        settings: Application settings (loaded automatically if None).
        model_name: Model name for a single-model run (--model option).
        experiment_name: Experiment name (--ex option; taken from config.yml if None).
        output_dir: Output directory for results.
        resume_from: Result directory of an interrupted run to resume.

    Returns:
        Path to the result directory.
    """
    # Register models in the registry
    register_all()

    result_dir: Path | None = None
    if resume_from is not None:
        result_dir = _resolve_resume_dir(resume_from, output_dir)
        settings = AppSettings.model_validate_json(
            ResultLayout(result_dir).config_json.read_text(encoding="utf-8")
        )
        logger.info(f"Resuming experiment from: {result_dir}")
    elif settings is None:
        settings = get_settings()

    # Determine the models to run and the experiment name
    if model_name:
        # Single-model run
        model_names = [model_name]
        exp_name = model_name
    else:
        # Models of the experiment section; the name may be overridden
        exp = get_experiment_config(settings)
        model_names = exp.models
        exp_name = experiment_name or exp.name

    if not model_names:
        raise ValueError("No models specified")

    # Create the result directory
    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = build_result_dir_name(exp_name, timestamp, settings.data)
        result_dir = Path(output_dir) / dir_name
        result_dir.mkdir(parents=True, exist_ok=True)
        _save_settings_snapshot(settings, ResultLayout(result_dir))

    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Models: {model_names}")
    logger.info(f"Output: {result_dir}")
    logger.info(f"Parameter selection scope: {parameter_selection_scope(settings)}")

    trial_seeds_by_snr = prepare_trial_seeds(settings, result_dir)

    # Run each model
    all_search_results: list[pd.DataFrame] = []
    all_best_params: list[dict[str, Any]] = []
    all_eval_results: list[pd.DataFrame] = []

    for name in model_names:
        spec = get_model(name)
        logger.info(f"Running model: {spec.name}")

        # Run the experiment (parameter selection + evaluation)
        search_df, best_params, eval_df = run_model(
            settings,
            spec,
            result_dir,
            trial_seeds_by_snr=trial_seeds_by_snr,
        )

        # Add model information to the results
        search_df["model"] = spec.name
        eval_df["model"] = spec.name

        all_search_results.append(search_df)
        all_eval_results.append(eval_df)

        all_best_params.extend(best_params_rows(spec.name, best_params))

    # Combine and save the results
    layout = ResultLayout(result_dir)
    layout.training_dir.mkdir(exist_ok=True)

    # search_results.csv
    combined_search = pd.concat(all_search_results, ignore_index=True)
    combined_search.to_csv(layout.search_results_csv, index=False)

    # best_params.csv
    best_params_df = pd.DataFrame(all_best_params)
    if best_params_df.empty:
        best_params_df = pd.DataFrame(columns=["model", "snr", "seed"])
    best_params_df.to_csv(layout.best_params_csv, index=False)

    # eval_results.csv
    layout.evaluation_dir.mkdir(exist_ok=True)
    combined_eval = pd.concat(all_eval_results, ignore_index=True)
    combined_eval.to_csv(layout.eval_results_csv, index=False)
    _export_trial_summaries(combined_eval, layout)

    # Table of mean and standard error of computation time (solve_time)
    _export_timing_tables(layout)

    logger.info(f"Experiment completed: {result_dir}")
    return result_dir


def _resolve_resume_dir(resume_from: str, output_dir: str) -> Path:
    """Resolve the run to resume from an absolute/relative path or a result name."""
    requested = Path(resume_from).expanduser()
    candidates = [requested, Path(output_dir) / requested]
    for candidate in candidates:
        if ResultLayout(candidate).config_json.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Resume result directory not found: {resume_from}")


def _export_trial_summaries(eval_df: pd.DataFrame, layout: ResultLayout) -> None:
    """Save failure rates and the evaluation results and summary on commonly successful trials."""
    layout.evaluation_dir.mkdir(parents=True, exist_ok=True)
    layout.tables_dir.mkdir(parents=True, exist_ok=True)

    success_summary = build_trial_success_summary(eval_df)
    success_summary.to_csv(layout.trial_success_summary_csv, index=False)

    common_df = select_common_success_trials(eval_df)
    common_df.to_csv(layout.common_success_eval_csv, index=False)
    common_summary = build_common_success_summary(common_df)
    common_summary.to_csv(layout.common_success_summary_csv, index=False)
    logger.info(
        "Trial summaries saved: "
        f"{layout.trial_success_summary_csv}, {layout.common_success_summary_csv}"
    )


def _export_timing_tables(layout: ResultLayout) -> None:
    """Save the solve_time summary (read by ``table``) and the SNR x method display table."""
    eval_path = layout.eval_results_csv
    if not eval_path.exists():
        raise FileNotFoundError(f"eval_results.csv not found: {eval_path}")
    eval_df = pd.read_csv(eval_path)
    layout.tables_dir.mkdir(parents=True, exist_ok=True)
    build_timing_summary(eval_df).to_csv(layout.timing_summary_csv, index=False)
    build_timing_display_table(eval_df).to_csv(layout.timing_table_csv, index=False)
    logger.info(f"Timing summary saved to: {layout.timing_summary_csv}")
    logger.info(f"Timing table saved to: {layout.timing_table_csv}")
