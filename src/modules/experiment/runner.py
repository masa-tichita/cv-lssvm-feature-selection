"""Per-model experiment runner: parameter selection and evaluation in one phase.

Step 1 (parameter selection) trains every parameter candidate on each trial and
records the objective; Step 2 (evaluation) retrains with the selected parameter
when needed and evaluates on the test data. Both steps are checkpointed so that
an interrupted run can be resumed.
"""

from pathlib import Path
from typing import Any, cast

import pandas as pd
from loguru import logger

from modules.data.factory import (
    SyntheticDataset,
    build_run_data_config,
    create_dataset,
)
from modules.experiment.checkpoint import (
    ModelCheckpoint,
    canonical_params,
    evaluation_key,
    search_key,
)
from modules.experiment.layout import ResultLayout
from modules.experiment.trials import TrialSeedsBySNR
from modules.models.base import ModelSpec, expand_param_grid
from modules.settings import AppSettings

BestParamsKey = float | tuple[float, int]
BestParams = dict[BestParamsKey, dict[str, Any]]


def parameter_selection_scope(settings: AppSettings) -> str:
    """Return the scope used for hyperparameter selection."""
    if settings.experiment is None:
        return "per_seed"
    return settings.experiment.parameter_selection_scope


def _select_best_params(
    spec: Any,
    search_df: pd.DataFrame,
    scope: str,
) -> BestParams:
    """Select the best parameters per SNR or per seed from the search results."""
    successful_df = search_df
    if "run_status" in search_df.columns:
        successful_df = search_df[search_df["run_status"] == "success"]
    successful_df = successful_df.dropna(subset=["objective"])
    if successful_df.empty:
        return {}

    if scope == "per_snr":
        return cast(BestParams, spec.select_best(successful_df))
    if scope != "per_seed":
        raise ValueError(f"Unknown parameter selection scope: {scope}")

    best_params: BestParams = {}
    for seed_value, seed_df in successful_df.groupby("seed", sort=False):
        seed: Any = seed_value
        for snr, params in spec.select_best(seed_df).items():
            best_params[(float(snr), int(seed))] = params
    return best_params


def _best_params_for(
    best_params: BestParams,
    snr: float,
    seed: int,
    scope: str,
) -> dict[str, Any] | None:
    """Return the best parameters for the given experiment instance."""
    key: BestParamsKey = (float(snr), int(seed)) if scope == "per_seed" else float(snr)
    return best_params.get(key)


def _needs_retrain(spec: ModelSpec) -> bool:
    """Return whether the model must be retrained after parameter selection.

    proposed_hl/ls: no retraining; the cross-validation structure is the model itself.
    logreg_aic/bic: no hyperparameters; the Step 1 model is already
        trained on the full X_train and is reused.
    Others: the model is retrained on X_train with the selected parameter
        (svm_cv re-solves its MIO with the selected C).
    """
    no_retrain = {"proposed_hl", "proposed_ls", "logreg_aic", "logreg_bic"}
    return spec.name not in no_retrain


def _train_params(
    params: dict[str, Any],
    result_dir: Path,
    snr: float,
    seed: int,
) -> dict[str, Any]:
    """Parameters passed to ``spec.train``: the candidate plus the run context.

    The run context (result_dir, snr, seed) lets the Gurobi-based models write
    their logs under ``<result_dir>/training/gurobi_logs/<model>/snr=<snr>/seed=<seed>``;
    the other models ignore it.
    """
    return {**params, "run_dir": result_dir, "snr": snr, "seed": seed}


def best_params_rows(model: str, best_params: BestParams) -> list[dict[str, Any]]:
    """Convert the selected parameters of one model to rows of best_params.csv."""
    rows: list[dict[str, Any]] = []
    for key, params in best_params.items():
        if isinstance(key, tuple):
            snr, seed = key
            rows.append({"model": model, "snr": snr, "seed": seed, **params})
        else:
            rows.append({"model": model, "snr": key, **params})
    return rows


def run_model(
    settings: AppSettings,
    spec: ModelSpec,
    result_dir: Path,
    trial_seeds_by_snr: TrialSeedsBySNR,
) -> tuple[pd.DataFrame, BestParams, pd.DataFrame]:
    """Run the experiment for a single model.

    Step 1: parameter selection
        - Generate data for each seed.
        - Train with each parameter and record the objective / CV score.
        - By default, select the best parameter for each seed.
        - Only with per_snr, select one value per SNR by averaging over seeds.

    Step 2: evaluation (per seed)
        - Retrain on the full X_train with the selected parameter
          (except proposed_hl/ls and logreg_aic/bic).
        - Evaluate on X_test.
        - Report mean ± standard deviation over seeds.

    Returns:
        (search_results_df, best_params, eval_results_df)
    """
    data = settings.data
    param_grid = spec.get_param_grid(settings)
    param_combinations = expand_param_grid(param_grid)

    snr_list = data.snr_list
    selection_scope = parameter_selection_scope(settings)

    model_desc = spec.name
    logger.info(f"Running experiment for {model_desc}")
    logger.info(f"SNR: {snr_list}, seeds by SNR: {trial_seeds_by_snr}")
    logger.info(f"Param grid: {param_grid}")

    # Step 1: parameter selection
    checkpoint = ModelCheckpoint(result_dir, model_desc)
    search_results: list[dict[str, Any]] = list(checkpoint.search_rows)
    completed_search_keys = checkpoint.completed_search_keys
    # Keep the datasets (reused in evaluation)
    datasets_by_snr_seed: dict[tuple[float, int], SyntheticDataset] = {}
    # Keep the solutions from parameter selection (reused by models without retraining)
    solutions_by_snr_seed_params: dict[tuple[float, int, str], Any] = {}

    n_trials = sum(len(trial_seeds_by_snr[float(snr)]) for snr in snr_list)
    total = n_trials * len(param_combinations)
    current = 0

    for snr in snr_list:
        for seed in trial_seeds_by_snr[float(snr)]:
            # Generate data
            data_config = build_run_data_config(data, snr=snr, seed=seed)
            dataset = create_dataset(data_config)
            datasets_by_snr_seed[(snr, seed)] = dataset

            for params in param_combinations:
                current += 1
                checkpoint_key = search_key(float(snr), int(seed), params)
                if checkpoint_key in completed_search_keys:
                    logger.info(
                        f"[{current}/{total}] checkpoint hit: "
                        f"snr={snr}, seed={seed}, params={params}"
                    )
                    continue
                logger.info(
                    f"[{current}/{total}] snr={snr}, seed={seed}, params={params}"
                )

                try:
                    solution, objective = spec.train(
                        X_train=dataset.X_train,
                        y_train=dataset.y_train,
                        params=_train_params(params, result_dir, snr, seed),
                        settings=settings,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Search failed: model={model_desc}, snr={snr}, "
                        f"seed={seed}, params={params}: {exc}"
                    )
                    failure_row = {
                        "_checkpoint_key": checkpoint_key,
                        "snr": snr,
                        "seed": seed,
                        **params,
                        "objective": float("nan"),
                        **_failure_fields("parameter_search", exc),
                    }
                    search_results.append(failure_row)
                    checkpoint.append_search(failure_row)
                    continue

                # Keep the solution (reused by models without retraining)
                if solution is not None:
                    solutions_by_snr_seed_params[(snr, seed, canonical_params(params))] = (
                        solution
                    )
                    if not _needs_retrain(spec):
                        spec.save(
                            solution,
                            checkpoint.solution_path(float(snr), int(seed), params),
                        )

                # Extract model-specific metrics (solution is None in cv_mode)
                if solution is not None:
                    train_metrics = spec.extract_train_metrics(solution)
                else:
                    train_metrics = {}

                result_row: dict[str, Any] = {
                    "_checkpoint_key": checkpoint_key,
                    "snr": snr,
                    "seed": seed,
                    **params,
                    "objective": objective,
                    **train_metrics,
                    **_success_fields(),
                }

                search_results.append(result_row)
                checkpoint.append_search(result_row)

    # Convert to a DataFrame
    search_df = checkpoint.search_frame()

    # Select the best parameters
    best_params = _select_best_params(spec, search_df, selection_scope)

    logger.info(f"Best params: {best_params}")

    # Step 2: evaluation
    eval_results: list[dict[str, Any]] = list(checkpoint.eval_rows)
    completed_evaluation_keys = checkpoint.completed_evaluation_keys
    needs_retrain = _needs_retrain(spec)

    layout = ResultLayout(result_dir)
    eval_total = n_trials
    eval_current = 0

    for snr in snr_list:
        snr_results: list[dict[str, Any]] = []

        for seed in trial_seeds_by_snr[float(snr)]:
            eval_checkpoint_key = evaluation_key(float(snr), int(seed))
            if eval_checkpoint_key in completed_evaluation_keys:
                eval_current += 1
                logger.info(
                    f"[Eval {eval_current}/{eval_total}] checkpoint hit: "
                    f"snr={snr}, seed={seed}"
                )
                continue
            params = _best_params_for(
                best_params,
                snr=float(snr),
                seed=int(seed),
                scope=selection_scope,
            )
            if params is None:
                logger.warning(
                    f"SNR={snr}, seed={seed}: No best params found; "
                    "recording failed trial"
                )
                failure_row = {
                    "_checkpoint_key": eval_checkpoint_key,
                    "_checkpoint_role": "primary",
                    "snr": snr,
                    "seed": seed,
                    **_search_failure_for_trial(search_df, snr, seed),
                }
                eval_results.append(failure_row)
                checkpoint.append_evaluation([failure_row])
                continue
            eval_current += 1
            logger.info(
                f"[Eval {eval_current}/{eval_total}] snr={snr}, seed={seed}, params={params}"
            )

            dataset = datasets_by_snr_seed[(snr, seed)]

            try:
                if needs_retrain:
                    # Retrain with the selected parameter
                    solution, _ = spec.train(
                        X_train=dataset.X_train,
                        y_train=dataset.y_train,
                        params=_train_params(params, result_dir, snr, seed),
                        settings=settings,
                    )
                else:
                    # Reuse the solution from parameter selection
                    solution = solutions_by_snr_seed_params.get(
                        (snr, seed, canonical_params(params))
                    )
                    if solution is None:
                        solution = spec.load(
                            checkpoint.solution_path(float(snr), int(seed), params)
                        )

                # Save the model
                spec.save(solution, layout.model_dir(spec.name, snr, seed))

                # Evaluate
                eval_result = spec.evaluate(
                    solution=solution,
                    X_test=dataset.X_test,
                    y_test=dataset.y_test,
                    true_support=dataset.true_support,
                    settings=settings,
                )
            except Exception as exc:
                logger.warning(
                    f"Evaluation failed: model={model_desc}, snr={snr}, "
                    f"seed={seed}, params={params}: {exc}"
                )
                failure_row = {
                    "_checkpoint_key": eval_checkpoint_key,
                    "_checkpoint_role": "primary",
                    "snr": snr,
                    "seed": seed,
                    **params,
                    **_failure_fields("evaluation", exc),
                }
                eval_results.append(failure_row)
                checkpoint.append_evaluation([failure_row])
                continue

            result_row = {
                "_checkpoint_key": eval_checkpoint_key,
                "_checkpoint_role": "primary",
                "snr": snr,
                "seed": seed,
                **params,
                **eval_result,
                **_success_fields(),
            }
            eval_results.append(result_row)
            snr_results.append(result_row)
            checkpoint.append_evaluation([result_row])

        # Log per-SNR statistics
        if snr_results:
            snr_df = pd.DataFrame(snr_results)
            logger.info(
                f"SNR={snr}: "
                f"n_selected={snr_df['n_selected'].mean():.1f}±{snr_df['n_selected'].std():.1f}, "
                f"model_f1={snr_df['model_f1'].mean():.3f}±{snr_df['model_f1'].std():.3f}"
            )

    eval_df = checkpoint.evaluation_frame()
    return search_df, best_params, eval_df


def _success_fields() -> dict[str, Any]:
    return {
        "run_status": "success",
        "failure_stage": None,
        "error_type": None,
        "error_message": None,
    }


def _failure_fields(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "run_status": "failed",
        "failure_stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _search_failure_for_trial(
    search_df: pd.DataFrame,
    snr: float,
    seed: int,
) -> dict[str, Any]:
    trial_rows = search_df[(search_df["snr"] == snr) & (search_df["seed"] == seed)]
    failed_rows = trial_rows[trial_rows.get("run_status") == "failed"]
    if not failed_rows.empty:
        row = failed_rows.iloc[0]
        return {
            "run_status": "failed",
            "failure_stage": "parameter_search",
            "error_type": row.get("error_type"),
            "error_message": row.get("error_message"),
        }
    return {
        "run_status": "failed",
        "failure_stage": "parameter_selection",
        "error_type": "NoValidParameters",
        "error_message": "No successful parameter candidate was available.",
    }

