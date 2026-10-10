"""Method keys, display labels, and value formatting used by the paper."""

# (method key in eval_results.csv, label used in the paper), in paper order.
METHODS: list[tuple[str, str]] = [
    ("proposed_hl", "Proposed-HL"),
    ("proposed_ls", "Proposed-LS"),
    ("svm_cv", "SVM-CV"),
    ("svm_l1", "SVM-L1"),
    ("svm_rfe", "SVM-RFE"),
    ("logreg_aic", "LogReg-AIC"),
    ("logreg_bic", "LogReg-BIC"),
]

METHOD_KEYS: list[str] = [key for key, _ in METHODS]
METHOD_LABELS: dict[str, str] = dict(METHODS)


def format_snr(snr: float) -> str:
    """Format an SNR value for tables and file names: 1.0 -> "1.0", 0.25 -> "0.25"."""
    snr = float(snr)
    if snr.is_integer():
        return f"{snr:.1f}"
    return f"{snr:g}"
