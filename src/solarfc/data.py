"""NSRDB Himawari PSM v3 loading.

File layout (NREL CSV export):

    row 0  metadata field names
    row 1  metadata values (site id, lat/lon, elevation, timezone, units)
    row 2  column headers
    row 3+ data

Timestamps are assembled from the Year/Month/Day/Hour/Minute columns and are
**UTC**. All modelling is done in UTC; local time is used only for plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_YEARS, NSRDB_DIR, STEP_MINUTES, SITES_BY_KEY

__all__ = ["load_site_year", "load_site", "FILL_FLAG_LABELS"]

#: NSRDB fill-flag semantics, for data-quality reporting.
FILL_FLAG_LABELS: dict[int, str] = {
    0: "OK",
    1: "Missing image",
    2: "Low irradiance",
    3: "Exceeds clearsky",
    4: "Missing cloud properties",
    5: "Rayleigh violation",
}

_TIME_COLS = ["Year", "Month", "Day", "Hour", "Minute"]


def load_site_year(site: str, year: int, data_dir=None) -> pd.DataFrame:
    """Load one ``{site}_{year}.csv`` into a UTC-indexed frame."""
    if site not in SITES_BY_KEY:
        raise KeyError(f"unknown site {site!r}; expected one of {sorted(SITES_BY_KEY)}")

    base = NSRDB_DIR if data_dir is None else data_dir
    path = base / f"{site}_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"NSRDB file not found: {path}")

    df = pd.read_csv(path, skiprows=2, low_memory=False)

    missing = [c for c in _TIME_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing time columns: {missing}")

    df.index = pd.to_datetime(df[_TIME_COLS], utc=True)
    df.index.name = "timestamp"
    df = df.drop(columns=_TIME_COLS)

    return df.sort_index()


def load_site(site: str, years=ALL_YEARS, data_dir=None) -> pd.DataFrame:
    """Load and concatenate several years for one site.

    Raises on duplicate timestamps: silently dropping them would corrupt the
    lag features and the ramp-window arithmetic that assume a uniform grid.
    """
    frames = [load_site_year(site, y, data_dir=data_dir) for y in years]
    df = pd.concat(frames).sort_index()

    dupes = int(df.index.duplicated().sum())
    if dupes:
        raise ValueError(f"{site}: {dupes} duplicate timestamps across {list(years)}")

    return df


def continuity_report(df: pd.DataFrame, step_minutes: int = STEP_MINUTES) -> dict:
    """Check the index is a uniform grid at ``step_minutes``.

    Every lag feature, every persistence baseline and every ramp window assumes
    evenly spaced samples. A gap silently shifts those windows and corrupts the
    results, so this is asserted rather than assumed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(df.index).__name__}")

    deltas = df.index.to_series().diff().dt.total_seconds().div(60).dropna()
    expected = pd.date_range(
        df.index.min(), df.index.max(), freq=f"{step_minutes}min", tz=df.index.tz
    )

    return {
        "rows": len(df),
        "expected_rows": len(expected),
        "missing_rows": len(expected) - len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "modal_step_min": float(deltas.mode().iloc[0]) if not deltas.empty else np.nan,
        "max_gap_min": float(deltas.max()) if not deltas.empty else np.nan,
        "irregular_steps": int((deltas != step_minutes).sum()),
    }
