"""``figures`` command: AUC / F1 / subset-size panels for each SNR of one result directory."""

from pathlib import Path

import pandas as pd
from loguru import logger

from modules.evaluation.plots import (
    apply_paper_style,
    panel_file_name,
    panel_values,
    render_panel,
)
from modules.experiment.layout import REFIT_SUMMARY_CSV, ResultLayout
from utils.labels import METHOD_KEYS


def build_figures(result_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Write the AUC, F1, and subset-size panels for each SNR of one result directory."""
    layout = ResultLayout(Path(result_dir))
    out = Path(output_dir) if output_dir is not None else layout.figures_dir
    apply_paper_style()
    theta = int(layout.read_config()["data"]["informative"])

    common = pd.read_csv(layout.common_success_summary_csv)
    refit = pd.read_csv(layout.refit_dir / REFIT_SUMMARY_CSV)

    for snr in sorted(common["snr"].unique()):
        for metric, (mean, se) in panel_values(common, refit, snr).items():
            if mean.isna().any():
                missing = [k for k, v in zip(METHOD_KEYS, mean.isna(), strict=True) if v]
                logger.warning(f"SNR={snr}, {metric}: missing methods {missing}")
            render_panel(
                mean.fillna(0.0).tolist(),
                se.fillna(0.0).tolist(),
                metric,
                theta,
                out / panel_file_name(float(snr), metric),
            )
    logger.info(f"Figures saved to: {out}")
    return out
