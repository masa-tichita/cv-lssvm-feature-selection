"""``refit`` command: common L2-SVM refit on the saved feature subsets of a result directory."""

import json
import time
from pathlib import Path

from loguru import logger

from modules.evaluation.refit import (
    collect_supports,
    refit_protocol,
    refit_supports,
    summarize_refit,
)
from modules.experiment.layout import (
    REFIT_PROTOCOL_JSON,
    REFIT_RESULTS_CSV,
    REFIT_SUMMARY_CSV,
    ResultLayout,
)
from modules.settings import RefitSVMSettings


def run_common_refit(result_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Run the common L2-SVM refit on all saved subsets of one result directory.

    Writes ``classification_results.csv`` (per trial), ``classification_summary.csv``
    (mean and standard error of the test AUC per method and SNR), and
    ``protocol.json`` to ``output_dir`` (default: ``<result_dir>/refit``).
    """
    layout = ResultLayout(Path(result_dir))
    out = Path(output_dir) if output_dir is not None else layout.refit_dir
    out.mkdir(parents=True, exist_ok=True)

    config = layout.read_config()
    refit_settings = RefitSVMSettings.model_validate(config["models"]["refit_svm"])
    records = collect_supports(layout.root)
    if not records:
        raise ValueError(f"No saved feature subsets found in {layout.root}")

    protocol = refit_protocol(refit_settings, run_count=len(records))
    (out / REFIT_PROTOCOL_JSON).write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    start = time.perf_counter()
    frame = refit_supports(records, config["data"], refit_settings)
    frame.to_csv(out / REFIT_RESULTS_CSV, index=False)
    summarize_refit(frame).to_csv(out / REFIT_SUMMARY_CSV, index=False)
    logger.info(
        f"Common refit completed: {len(frame)} runs in "
        f"{time.perf_counter() - start:.1f}s -> {out}"
    )
    return out
