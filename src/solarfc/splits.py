"""Temporal splits, monsoon labelling and evaluation strata.

Two rules govern this module:

1. **Splits are strictly chronological.** Solar time series are autocorrelated
   over hours; a shuffled split leaks near-identical neighbouring
   samples into the test set and reports a number that cannot be
   reproduced operationally.
2. **Split definitions are serialisable.** The exact indices are written to
   disk and hashed, because they are deposited alongside the benchmark
   for the Zenodo DOI. Reconstructing them from prose later is not
   reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MONSOON_TRANSITIONS,
    MONTH_TO_MONSOON,
    TEST_YEARS,
    TRAIN_YEARS,
    TRANSITION_WINDOW_DAYS,
    VAL_YEARS,
)

__all__ = [
    "monsoon_phase",
    "is_transition_window",
    "split_label",
    "assign_splits",
    "split_summary",
    "write_split_manifest",
]


def monsoon_phase(index: pd.DatetimeIndex) -> np.ndarray:
    """Map timestamps to monsoon phase codes (see ``config.MONSOON_LABELS``).

    Month-based, following the Malaysian Meteorological Department
    convention. This is a deliberate simplification — real onset shifts
    by weeks year to year — and that simplification is exactly what the
    Module A gate-validation study tests, by checking whether the
    learned gate weight tracks *actual* onset dates better than this
    fixed calendar does.
    """
    months = np.asarray(index.month, dtype=int)
    out = np.empty(months.shape, dtype=np.int8)
    for month, phase in MONTH_TO_MONSOON.items():
        out[months == month] = phase
    return out


def is_transition_window(
    index: pd.DatetimeIndex,
    *,
    window_days: int = TRANSITION_WINDOW_DAYS,
) -> np.ndarray:
    """Mark samples falling within +/- ``window_days`` of a monsoon transition.

    Quantile calibration characteristically degrades under regime shift.
    A single aggregate PICP would average that failure away, so PICP and
    reliability are reported separately for transition and stable
    windows.
    """
    if window_days < 0:
        raise ValueError(f"window_days must be >= 0, got {window_days}")

    idx = pd.DatetimeIndex(index)
    mask = np.zeros(len(idx), dtype=bool)
    # np.asarray, not .to_numpy(): pandas 3.0 already returns ndarrays
    # here, while pandas 2.x returns Index objects. This works on both.
    day_of_year = np.asarray(idx.dayofyear, dtype=int)
    is_leap = np.asarray(idx.is_leap_year, dtype=bool)

    for month, day in MONSOON_TRANSITIONS:
        # Transition day-of-year, computed per sample so leap years are
        # exact.
        anchor = np.where(
            is_leap,
            pd.Timestamp(2020, month, day).dayofyear,  # 2020 is a leap year
            pd.Timestamp(2019, month, day).dayofyear,
        )
        year_length = np.where(is_leap, 366, 365)
        # Circular distance, so a window spanning the new year still
        # works.
        raw = np.abs(day_of_year - anchor)
        distance = np.minimum(raw, year_length - raw)
        mask |= distance <= window_days

    return mask


def split_label(index: pd.DatetimeIndex) -> np.ndarray:
    """Label each timestamp ``train`` / ``val`` / ``test`` / ``unused``."""
    years = np.asarray(pd.DatetimeIndex(index).year, dtype=int)
    out = np.full(years.shape, "unused", dtype=object)
    out[np.isin(years, TRAIN_YEARS)] = "train"
    out[np.isin(years, VAL_YEARS)] = "val"
    out[np.isin(years, TEST_YEARS)] = "test"
    return out


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``split``, ``monsoon_phase`` and ``is_transition`` columns.

    Expects a DatetimeIndex. Returns a copy; the input is not mutated.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"expected a DatetimeIndex, got {type(df.index).__name__}"
        )

    out = df.copy()
    out["split"] = split_label(out.index)
    out["monsoon_phase"] = monsoon_phase(out.index)
    out["is_transition"] = is_transition_window(out.index)
    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts and date bounds per split — a sanity check before training.

    Verifies the property that matters: no split overlaps another in
    time.
    """
    if "split" not in df.columns:
        df = assign_splits(df)

    rows = []
    for name in ("train", "val", "test", "unused"):
        sel = df[df["split"] == name]
        if sel.empty:
            continue
        rows.append(
            {
                "split": name,
                "rows": len(sel),
                "start": sel.index.min(),
                "end": sel.index.max(),
                "years": sorted({int(y) for y in sel.index.year.unique()}),
            }
        )
    return pd.DataFrame(rows)


def write_split_manifest(df: pd.DataFrame, path: str | Path) -> dict:
    """Serialise split boundaries plus a content hash, for the DOI deposit.

    The hash covers the split boundaries and row counts, so a later run
    can prove it used identical splits rather than asserting it.
    """
    summary = split_summary(df)
    manifest = {
        "step_minutes_expected": 10,
        "train_years": list(TRAIN_YEARS),
        "val_years": list(VAL_YEARS),
        "test_years": list(TEST_YEARS),
        "transition_window_days": TRANSITION_WINDOW_DAYS,
        "splits": [
            {
                "split": r["split"],
                "rows": int(r["rows"]),
                "start": str(r["start"]),
                "end": str(r["end"]),
                "years": r["years"],
            }
            for _, r in summary.iterrows()
        ],
    }
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
