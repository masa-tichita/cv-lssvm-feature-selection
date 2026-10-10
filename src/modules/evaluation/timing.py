"""Computation-time summaries and tables.

All timing outputs are derived from one summary: the mean, standard deviation,
and standard error of ``solve_time`` per method and SNR
(``build_timing_summary``; saved as ``tables/timing_summary.csv``).
From this summary,

* ``build_timing_display_table`` gives the per-run SNR x method display table
  (``tables/timing_table.csv``), and
* ``combine_timing_summaries`` / ``render_timing_latex`` give the paper's
  computation-time table over several conditions (Table 1).
"""

from typing import Any

import pandas as pd

from utils.labels import METHOD_KEYS, METHOD_LABELS, METHODS, format_snr

TIMING_SUMMARY_COLUMNS = [
    "model",
    "snr",
    "time_n",
    "solve_time_mean",
    "solve_time_std",
    "solve_time_se",
]


def build_timing_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize computation time by model and SNR (mean, SD, standard error)."""
    _check_columns(eval_df)
    timing_df = eval_df[["model", "snr", "solve_time"]].copy()
    timing_df["solve_time"] = pd.to_numeric(timing_df["solve_time"], errors="coerce")
    timing_df = timing_df.dropna(subset=["solve_time"])
    if timing_df.empty:
        return pd.DataFrame(columns=TIMING_SUMMARY_COLUMNS)

    summary = (
        timing_df.groupby(["model", "snr"], as_index=False)["solve_time"]
        .agg(time_n="count", solve_time_mean="mean", solve_time_std="std")
        .sort_values(["model", "snr"], ignore_index=True)
    )
    summary["solve_time_se"] = summary["solve_time_std"] / summary["time_n"].pow(0.5)
    return summary[TIMING_SUMMARY_COLUMNS]


def build_timing_display_table(
    eval_df: pd.DataFrame,
    decimals: int = 2,
    min_display: float = 0.01,
) -> pd.DataFrame:
    """SNR x method display table ("mean\\n±SE") of one result directory."""
    summary = build_timing_summary(eval_df).set_index(["model", "snr"])
    snr_values = sorted(eval_df["snr"].dropna().unique())
    models = _ordered_models(eval_df)

    rows: list[dict[str, str]] = []
    for snr in snr_values:
        row: dict[str, str] = {"SNR": format_snr(float(snr))}
        for model in models:
            key = (model, snr)
            if key in summary.index:
                mean = float(summary.at[key, "solve_time_mean"])
                se = float(summary.at[key, "solve_time_se"])
            else:
                mean = se = float("nan")
            row[METHOD_LABELS.get(model, model)] = _format_display_cell(
                mean, se, decimals=decimals, min_display=min_display
            )
        rows.append(row)
    return pd.DataFrame(rows)


def combine_timing_summaries(
    conditions: list[tuple[dict[str, Any], pd.DataFrame]],
) -> pd.DataFrame:
    """Stack the timing summaries of several conditions (long format, paper methods only).

    Args:
        conditions: (``data`` section of config.json, timing summary) per condition.
    """
    frames = []
    for data, summary in conditions:
        summary = summary.copy()
        summary["p"] = int(data["features"])
        summary["theta"] = int(data["informative"])
        summary["rho"] = float(data["rho"])
        frames.append(summary)
    table = pd.concat(frames, ignore_index=True)
    table = table[table["model"].isin(METHOD_LABELS)].copy()
    table["method"] = table["model"].map(METHOD_LABELS)
    return table


def render_timing_latex(table: pd.DataFrame, decimals: int = 2) -> str:
    """Render the combined timing table as the paper's LaTeX table (Table 1)."""
    snrs = sorted(table["snr"].unique())
    conditions = sorted(set(zip(table["p"], table["theta"], table["rho"], strict=True)))
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Mean computation times (in seconds), with standard errors in parentheses, "
        r"for each combination of \(p\), \(\theta\), \(\rho\), and SNR.}",
        r"\label{tab:computation-time}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}lccc" + "r" * len(snrs) + "@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\(p\)} & "
        r"\multirow{2}{*}{\(\theta\)} & \multirow{2}{*}{\(\rho\)} & "
        rf"\multicolumn{{{len(snrs)}}}{{c}}{{\textbf{{SNR}}}} \\",
        rf"\cmidrule(lr){{5-{4 + len(snrs)}}}",
        " & & & & " + " & ".join(f"{snr:.2f}" for snr in snrs) + r" \\",
        r"\midrule",
    ]
    for m_idx, (model, label) in enumerate(METHODS):
        for c_idx, (p, theta, rho) in enumerate(conditions):
            cells = []
            for snr in snrs:
                row = table[
                    (table["model"] == model)
                    & (table["p"] == p)
                    & (table["rho"] == rho)
                    & (table["snr"] == snr)
                ]
                if len(row) != 1:
                    cells.append("--")
                    continue
                cells.append(
                    _format_latex_cell(
                        float(row.iloc[0]["solve_time_mean"]),
                        float(row.iloc[0]["solve_time_se"]),
                        decimals,
                    )
                )
            head = rf"\multirow{{{len(conditions)}}}{{*}}{{{label}}}" if c_idx == 0 else ""
            lines.append(f"{head} & {p} & {theta} & {rho:.2f} & " + " & ".join(cells) + r" \\")
        if m_idx < len(METHODS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _check_columns(eval_df: pd.DataFrame) -> None:
    required = {"snr", "model", "solve_time"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_results.csv missing columns: {sorted(missing)}")


def _ordered_models(eval_df: pd.DataFrame) -> list[str]:
    existing = set(eval_df["model"].dropna().unique())
    ordered = [model for model in METHOD_KEYS if model in existing]
    remaining = sorted(str(model) for model in existing - set(ordered))
    return ordered + remaining


def _format_display_cell(
    mean: float,
    se: float,
    decimals: int,
    min_display: float,
) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(se):
        se = 0.0
    if mean < min_display and se < min_display:
        return f"<{min_display:.{decimals}f}"
    return f"{mean:.{decimals}f}\n±{se:.{decimals}f}"


def _format_latex_cell(
    mean: float, se: float, decimals: int = 2, min_display: float = 0.01
) -> str:
    mean_str = (
        f"{{\\ensuremath{{<{min_display:.{decimals}f}}}}}"
        if mean < min_display
        else f"{mean:.{decimals}f}"
    )
    return f"{mean_str} (\\(\\pm\\){se:.{decimals}f})"
