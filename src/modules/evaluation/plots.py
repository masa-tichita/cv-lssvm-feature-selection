"""Figure panels (AUC / F1 / subset size) of the paper.

Each panel is a horizontal bar chart of the per-method mean with standard-error bars:

* AUC: common L2-SVM refit (``<result_dir>/refit/classification_summary.csv``).
* F1 and subset size: feature-selection results on the trials where every
  method succeeded (``<result_dir>/tables/common_success_summary.csv``).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils.labels import METHOD_KEYS, METHODS, format_snr  # noqa: E402

PANEL_METRICS = ("model_auc", "feature_f1", "n_selected")
_METRIC_FILE_KEY = {"model_auc": "auc", "feature_f1": "f1", "n_selected": "n_selected"}


def apply_paper_style() -> None:
    """Set the font used by the paper's figures."""
    plt.rcParams["font.family"] = ["Arial", "sans-serif"]


def panel_file_name(snr: float, metric: str) -> str:
    """File name of one panel, e.g. ``comparison_snr_1_0_auc_en.png``."""
    tag = format_snr(snr).replace(".", "_")
    return f"comparison_snr_{tag}_{_METRIC_FILE_KEY[metric]}_en.png"


def panel_values(
    common_summary: pd.DataFrame,
    refit_summary: pd.DataFrame,
    snr: float,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Return (mean, SE) per method, in paper order, for each panel metric at one SNR."""
    rows = common_summary[common_summary["snr"] == snr].set_index("model").reindex(METHOD_KEYS)
    auc = refit_summary[refit_summary["snr"] == snr].set_index("model").reindex(METHOD_KEYS)
    return {
        "model_auc": (auc["refit_auc"], auc["refit_auc_se"]),
        "feature_f1": (rows["feature_f1_mean"], rows["feature_f1_se"]),
        "n_selected": (rows["n_selected_mean"], rows["n_selected_se"]),
    }


def render_panel(
    means: list[float],
    ses: list[float],
    metric: str,
    theta: int,
    output_path: Path,
) -> None:
    """Draw one horizontal bar panel in the paper's style."""
    labels = [label for _, label in METHODS]
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(
        range(len(labels)),
        means,
        xerr=ses,
        capsize=3,
        color=[plt.colormaps["tab10"](i) for i in range(len(labels))],
        alpha=1.0 if metric == "n_selected" else 0.8,
        height=0.75,
    )
    fmt = ".1f" if metric == "n_selected" else ".2f"
    for bar, mean, se in zip(bars, means, ses, strict=True):
        ax.text(
            bar.get_width() + se + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:{fmt}}",
            ha="left",
            va="center",
            fontsize=30,
        )
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=30)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=28)
    if metric == "n_selected":
        ax.set_xlim(0, max(1.0, max(max(means), theta) * 1.4))
        ax.axvline(theta, color="0.35", linestyle="-", linewidth=2.0, alpha=0.55, zorder=0)
    else:
        ax.set_xlim(0, 1.15)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
