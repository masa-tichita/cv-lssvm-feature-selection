"""``table`` command: computation-time table (Table 1) from the result directories of the conditions."""

from pathlib import Path

import pandas as pd
from loguru import logger

from modules.evaluation.timing import combine_timing_summaries, render_timing_latex
from modules.experiment.layout import ResultLayout


def build_timing_table(
    result_dirs: list[str | Path],
    output_dir: str | Path = "paper_outputs",
    decimals: int = 2,
) -> Path:
    """Combine the timing summaries of several result directories into Table 1.

    Writes ``timing_table.csv`` (long format) and ``timing_table.tex`` (LaTeX table).
    """
    conditions = []
    for result_dir in result_dirs:
        layout = ResultLayout(Path(result_dir))
        summary = pd.read_csv(layout.timing_summary_csv)
        conditions.append((layout.read_config()["data"], summary))
    table = combine_timing_summaries(conditions)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.sort_values(["p", "rho", "snr"]).to_csv(out / "timing_table.csv", index=False)
    tex_path = out / "timing_table.tex"
    tex_path.write_text(render_timing_latex(table, decimals), encoding="utf-8")
    logger.info(f"Timing table saved to: {tex_path}")
    return tex_path
