"""Checkpointing that incrementally saves the search and evaluation of a run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from modules.experiment.layout import ResultLayout


def canonical_params(params: dict[str, Any]) -> str:
    """Convert search parameters to an order-independent string."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def search_key(snr: float, seed: int, params: dict[str, Any]) -> str:
    """Return a key that uniquely identifies one search run."""
    return f"{float(snr):.17g}|{int(seed)}|{canonical_params(params)}"


def evaluation_key(snr: float, seed: int) -> str:
    """Return a key that uniquely identifies one evaluation trial."""
    return f"{float(snr):.17g}|{int(seed)}"


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return normalized or "model"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write atomically so that an interrupted run leaves no broken CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


class ModelCheckpoint:
    """Track the progress of a single model (e.g., proposed_hl)."""

    def __init__(self, result_dir: Path, model_id: str) -> None:
        self.root = ResultLayout(result_dir).checkpoints_dir / _safe_name(model_id)
        self.search_path = self.root / "search_results.csv"
        self.eval_path = self.root / "eval_results.csv"
        self.search_rows = _read_csv(self.search_path).to_dict("records")
        self.eval_rows = _read_csv(self.eval_path).to_dict("records")

    @property
    def completed_search_keys(self) -> set[str]:
        return {
            str(row["_checkpoint_key"])
            for row in self.search_rows
            if pd.notna(row.get("_checkpoint_key"))
        }

    @property
    def completed_evaluation_keys(self) -> set[str]:
        return {
            str(row["_checkpoint_key"])
            for row in self.eval_rows
            if row.get("_checkpoint_role") == "primary"
            and pd.notna(row.get("_checkpoint_key"))
        }

    def append_search(self, row: dict[str, Any]) -> None:
        self.search_rows.append(row)
        _atomic_write_csv(self.search_rows, self.search_path)

    def append_evaluation(self, rows: list[dict[str, Any]]) -> None:
        self.eval_rows.extend(rows)
        _atomic_write_csv(self.eval_rows, self.eval_path)

    def solution_path(
        self,
        snr: float,
        seed: int,
        params: dict[str, Any],
    ) -> Path:
        digest = hashlib.sha256(canonical_params(params).encode()).hexdigest()[:16]
        return (
            self.root
            / "search_solutions"
            / f"snr_{float(snr):g}"
            / f"seed_{int(seed)}"
            / digest
        )

    def search_frame(self) -> pd.DataFrame:
        return _drop_internal_columns(pd.DataFrame(self.search_rows))

    def evaluation_frame(self) -> pd.DataFrame:
        return _drop_internal_columns(pd.DataFrame(self.eval_rows))


def _drop_internal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    internal = [column for column in frame.columns if column.startswith("_checkpoint_")]
    return frame.drop(columns=internal, errors="ignore")
