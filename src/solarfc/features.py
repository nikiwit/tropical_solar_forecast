"""Observed-past feature engineering, shared by every model.

Scope
-----
This module builds the *encoder* side: everything the model may look at from
time ``t`` backwards, where ``t`` is the forecast origin. The *decoder* side --
what is legitimately knowable about ``t + h`` at issue time -- lives in
:mod:`solarfc.covariates` and is joined in by :mod:`solarfc.dataset`.

The split is not cosmetic. It is the only structural defence against
leakage in a direct multi-horizon setup, where the target sits ``h``
steps in the future and a single mis-signed shift would silently hand
the model the answer. Every column produced here is trailing by
construction, and every column joined from the known-future side carries
a ``kf_`` prefix, so a leakage audit is a string check rather than a
reading of the code.

Clear-sky reference
-------------------
Two different clear-sky GHI series are available and they are not
interchangeable:

``nsrdb_clearsky_ghi``
    NSRDB PSM v3's own clear-sky estimate, computed by NREL's radiative
    transfer using satellite-retrieved aerosol optical depth and ozone.
    Highly accurate, and unavailable to any real plant.

``kf_clearsky_ghi``
    pvlib Ineichen with the Linke turbidity climatology, from
    coordinates and timestamp alone. Computable on site, years ahead,
    and it is what SolarInfer reimplements in C++.

The clear-sky index target and the CSI features use **Ineichen**, in
both feature sets. Using NSRDB's series would make the clear-sky index
itself a satellite product, which would put satellite information into
the DEPLOYABLE set through the back door and quietly shrink the
deployable-vs-full gap that Contribution 4 reports. NSRDB's series is
retained as an ordinary input column in the FULL set only.

Note that this makes ``config.DEPLOYABLE_NSRDB_FEATURES`` -- which lists
``Clearsky GHI`` -- narrower in practice than it reads: see
:data:`DEPLOYABLE_DROP` below.

The clear-sky index at night
----------------------------
The clear-sky index is a ratio against an envelope that is zero at night, so it
is undefined for 53% of the record. That is not a nuisance to be patched over:
handled naively it removes whole horizons. Measured at Kuala Lumpur, requiring
the forecast *origin* to be daytime as well as the target loses 53% of rows at
6 h and 18 h and **100% at 12 h and 36 h** -- every daytime target at those
horizons has a night origin, 36 h being the Declared Daily Capacity submission.

One convention is applied throughout, matching
``baselines.smart_persistence(carry_overnight=True)`` so the package has
a single night rule rather than two:

    **A CSI-derived quantity carries its last computable value across
    the night.**

Rolling statistics are computed NaN-skip on the *raw* index first, so a
24-hour mean averages the real daytime samples in its window rather than
one value repeated seventy times, and only the result is carried.
``csi_age_steps`` records how many steps old the carried value is, so a
model can discount a stale one instead of being told a fresh observation
exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import clear_sky_index
from .config import SATELLITE_ONLY_FEATURES, STEP_MINUTES, Site, SITES_BY_KEY
from .covariates import solar_geometry

__all__ = [
    "NSRDB_RENAME",
    "LAG_STEPS_IRRADIANCE",
    "LAG_STEPS_METEO",
    "ROLLING_MEAN_STEPS",
    "ROLLING_STD_STEPS",
    "SATELLITE_ONLY_COLUMNS",
    "DEPLOYABLE_DROP",
    "FEATURE_SETS",
    "build_observed_past",
    "feature_columns",
    "assert_no_leakage",
]

# --------------------------------------------------------------------------
# Column naming
# --------------------------------------------------------------------------
#
# NSRDB ships title-case headers with spaces. Renaming once here means
# no downstream module has to quote 'Clearsky GHI' or remember which
# dataset a column came from.

NSRDB_RENAME: dict[str, str] = {
    "GHI": "ghi",
    "DNI": "dni",
    "DHI": "dhi",
    "Clearsky GHI": "nsrdb_clearsky_ghi",
    "Clearsky DNI": "nsrdb_clearsky_dni",
    "Clearsky DHI": "nsrdb_clearsky_dhi",
    "Temperature": "temp_c",
    "Dew Point": "dew_point_c",
    "Relative Humidity": "relative_humidity",
    "Pressure": "pressure_mbar",
    "Wind Speed": "wind_speed",
    "Wind Direction": "wind_direction",
    "Precipitable Water": "precipitable_water",
    "Surface Albedo": "surface_albedo",
    "Cloud Type": "cloud_type",
    "Fill Flag": "fill_flag",
    "Solar Zenith Angle": "nsrdb_solar_zenith",
    "Aerosol Optical Depth": "aod",
    "Alpha": "alpha",
    "Ozone": "ozone",
    "Asymmetry": "asymmetry",
}

#: Renamed equivalents of ``config.SATELLITE_ONLY_FEATURES``.
SATELLITE_ONLY_COLUMNS: tuple[str, ...] = tuple(
    NSRDB_RENAME[c] for c in SATELLITE_ONLY_FEATURES
)

#: Dropped from the DEPLOYABLE set on top of the satellite-only list.
#:
#: NSRDB's clear-sky series are radiative-transfer output driven by
#: retrieved aerosol and ozone, so they are satellite products even
#: though the plan's feature table lists 'Clearsky GHI' as deployable. A
#: site computes its own clear-sky envelope from geometry
#: (``kf_clearsky_ghi``), which is supplied to both feature sets, so
#: nothing is lost -- but leaving NSRDB's version in would understate
#: the gap the DEPLOYABLE set exists to measure.
#:
#: ``fill_flag`` is a retrieval-quality code, not a measurement, and is
#: dropped from both sets: it is metadata about the label, and in the
#: FULL set it would let the model detect exactly the timesteps where
#: its own target is unreliable.
DEPLOYABLE_DROP: tuple[str, ...] = (
    "nsrdb_clearsky_ghi",
    "nsrdb_clearsky_dni",
    "nsrdb_clearsky_dhi",
    "nsrdb_solar_zenith",
)

#: Dropped from every feature set, in both FULL and DEPLOYABLE.
ALWAYS_DROP: tuple[str, ...] = ("fill_flag",)

FEATURE_SETS: tuple[str, ...] = ("full", "deployable")

# --------------------------------------------------------------------------
# Lag and rolling windows
# --------------------------------------------------------------------------
#
# Taken directly from the plan's feature list, converted from wall-clock
# to steps at the native 10-minute resolution.

#: t-10min, t-30min, t-1h, t-3h, t-6h, t-24h.
LAG_STEPS_IRRADIANCE: tuple[int, ...] = (1, 3, 6, 18, 36, 144)

#: Meteorological drivers move slowly relative to irradiance, so a
#: sub-hourly lag set adds columns without adding information. 1 h and 6
#: h only.
LAG_STEPS_METEO: tuple[int, ...] = (6, 36)

#: 30 min, 3 h, 24 h.
ROLLING_MEAN_STEPS: tuple[int, ...] = (3, 18, 144)

#: 6 h. Rolling standard deviation is the variability proxy -- a
#: high-variance recent past is a convective sky, which is where the
#: long horizons fail.
ROLLING_STD_STEPS: tuple[int, ...] = (36,)

#: Window for the Module B deviation signal, in steps (144 = 24 h).
DELTA_CSI_WINDOW_STEPS = 144

#: Minimum valid samples in a CSI rolling window, as a fraction of the
#: window.
#:
#: CSI is NaN at night, so a full-window ``min_periods`` can never be
#: met and every CSI rolling statistic would be NaN -- which is what
#: silently emptied ``delta_csi``, Module B's input, before this was
#: caught. Daytime is ~47% of the record, so a quarter of the window is
#: a floor that real daytime coverage clears comfortably while still
#: rejecting a statistic built from one or two samples at a dawn
#: boundary.
CSI_ROLLING_MIN_FRACTION = 0.25

#: Base columns whose night-time NaN is carried forward rather than
#: dropped.
_NIGHT_CARRIED: tuple[str, ...] = ("csi",)

#: Variables receiving the full irradiance lag/rolling treatment.
_IRRADIANCE_BASE: tuple[str, ...] = ("ghi", "csi")

#: Variables receiving the shorter meteorological lag set.
_METEO_BASE: tuple[str, ...] = (
    "temp_c",
    "relative_humidity",
    "wind_speed",
    "era5_cloud_cover",
    "era5_precip_mm_h",
)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_observed_past(
    nsrdb: pd.DataFrame,
    site: Site | str,
    era5: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build every observed-past feature for one site, on the native grid.

    Parameters
    ----------
    nsrdb : DataFrame
        Raw NSRDB frame from :func:`solarfc.data.load_site`, UTC-indexed
        on a uniform 10-minute grid.
    site : Site or str
        Determines the Ineichen clear-sky reference.
    era5 : DataFrame, optional
        Upsampled ERA5 features from the site cache. Reindexed onto the
        NSRDB grid; missing timestamps become NaN rather than being
        interpolated again, since the upsampling decision was already
        made and recorded in :mod:`solarfc.era5`.

    Returns
    -------
    DataFrame
        UTC-indexed, one row per origin timestamp. Contains raw columns,
        the clear-sky index, lags, rolling statistics and the Module B
        deviation signal. Every column is a function of data at or
        before its own timestamp.

    Notes
    -----
    Rolling statistics over a fully-defined series (GHI) use ``min_periods``
    equal to the window, so a partial window yields NaN rather than a statistic
    quietly computed over fewer samples -- that would change what the feature
    means for the first day of the record without changing any metric.

    The clear-sky index cannot meet that, since night is NaN by
    construction, so its windows use ``CSI_ROLLING_MIN_FRACTION`` of the
    window instead and the result is carried across the night. See the
    module docstring.
    """
    if isinstance(site, str):
        site = SITES_BY_KEY[site]

    out = nsrdb.rename(columns=NSRDB_RENAME)
    keep = [c for c in NSRDB_RENAME.values() if c in out.columns]
    out = out[keep].copy()

    # Ineichen clear-sky: the deployable reference, and the CSI
    # denominator.
    geometry = solar_geometry(out.index, site)
    out["clearsky_ghi_ineichen"] = geometry["clearsky_ghi"].to_numpy()
    out["solar_zenith"] = geometry["solar_zenith"].to_numpy()
    out["cos_zenith"] = geometry["cos_zenith"].to_numpy()

    # Raw index: NaN at night by construction. Kept out of the feature
    # set and used only as the input to the rolling statistics, so those
    # average real observations rather than a carried value.
    csi_raw = pd.Series(
        clear_sky_index(
            out["ghi"].to_numpy(), out["clearsky_ghi_ineichen"].to_numpy()
        ),
        index=out.index,
        name="csi",
    )
    out["csi"] = csi_raw.ffill()
    out["csi_age_steps"] = _steps_since_observed(csi_raw.to_numpy())

    # Wind direction wraps at 360, which a tree splits badly and a
    # network cannot represent at all. Decompose to components; the raw
    # bearing is kept for interpretability but the components are what
    # carry the signal.
    if {"wind_speed", "wind_direction"}.issubset(out.columns):
        bearing = np.radians(out["wind_direction"].to_numpy())
        out["wind_u"] = -out["wind_speed"].to_numpy() * np.sin(bearing)
        out["wind_v"] = -out["wind_speed"].to_numpy() * np.cos(bearing)

    if era5 is not None:
        joined = era5.reindex(out.index)
        for column in joined.columns:
            out[column] = joined[column].to_numpy()

    # Lags read the carried series -- 'what was the sky doing' has an
    # answer at 03:00, and it is the last thing that was actually
    # observed.
    out = _add_lags(out)
    out = _add_rolling(out, raw={"csi": csi_raw})

    # Module B's signal, precomputed here so the TFT and the tree
    # baselines see an identical definition rather than two
    # implementations that drift.
    out["delta_csi"] = (
        out["csi"] - out[f"csi_roll_mean_{DELTA_CSI_WINDOW_STEPS}"]
    )

    out.attrs["site"] = site.key
    out.attrs["step_minutes"] = STEP_MINUTES
    return out


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Append lagged copies of the irradiance and meteorological drivers.

    A positive shift looks backwards, so ``ghi_lag_6`` at time ``t`` is
    the observation from ``t - 1 h``. There is no configuration that
    would make this look forwards, which is deliberate.
    """
    out = df.copy()
    new: dict[str, np.ndarray] = {}

    for column in _IRRADIANCE_BASE:
        if column not in out.columns:
            continue
        for lag in LAG_STEPS_IRRADIANCE:
            new[f"{column}_lag_{lag}"] = out[column].shift(lag).to_numpy()

    for column in _METEO_BASE:
        if column not in out.columns:
            continue
        for lag in LAG_STEPS_METEO:
            new[f"{column}_lag_{lag}"] = out[column].shift(lag).to_numpy()

    return out.assign(**new)


def _steps_since_observed(values) -> np.ndarray:
    """Steps elapsed since the last finite sample, 0 where the sample is finite.

    NaN before the first observation, so the leading edge is dropped
    rather than given a fabricated age.
    """
    finite = np.isfinite(np.asarray(values, dtype=float).ravel())
    position = np.arange(finite.size)
    last = np.maximum.accumulate(np.where(finite, position, -1))
    return np.where(last >= 0, (position - last).astype(float), np.nan)


def _add_rolling(
    df: pd.DataFrame, raw: dict[str, pd.Series] | None = None
) -> pd.DataFrame:
    """Append trailing rolling means and standard deviations.

    ``raw`` supplies an alternative input series for a base column. It
    exists for the clear-sky index: the statistic is computed on the raw
    index so it averages real daytime observations, then carried across
    the night. Computing it on the already-carried series would repeat
    the last value before sunset for the length of the night and pull a
    24-hour mean toward it.
    """
    out = df.copy()
    raw = raw or {}
    new: dict[str, np.ndarray] = {}

    for column in _IRRADIANCE_BASE:
        if column not in out.columns:
            continue
        series = raw.get(column, out[column])
        carried = column in _NIGHT_CARRIED

        def finish(rolled: pd.Series) -> np.ndarray:
            return (rolled.ffill() if carried else rolled).to_numpy()

        for window in ROLLING_MEAN_STEPS:
            floor = _min_periods(window, carried)
            new[f"{column}_roll_mean_{window}"] = finish(
                series.rolling(window, min_periods=floor).mean()
            )
        for window in ROLLING_STD_STEPS:
            floor = max(2, _min_periods(window, carried))
            new[f"{column}_roll_std_{window}"] = finish(
                series.rolling(window, min_periods=floor).std()
            )

    return out.assign(**new)


def _min_periods(window: int, carried: bool) -> int:
    """Minimum valid samples for a rolling window.

    A fully-defined series demands the whole window, so a partial
    statistic is never silently substituted for a complete one. A
    night-interrupted series cannot meet that and uses a fraction
    instead.
    """
    if not carried:
        return window
    return max(1, int(window * CSI_ROLLING_MIN_FRACTION))


# --------------------------------------------------------------------------
# Feature-set selection
# --------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame, feature_set: str = "full") -> list[str]:
    """Observed-past columns belonging to a feature set.

    ``full`` is every engineered column. ``deployable`` additionally
    removes the satellite-only retrievals and NSRDB's own clear-sky and
    zenith series, plus anything derived from them. The difference
    between the two is the headline number for an operator: it is what a
    plant loses by not having a satellite.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}"
        )

    banned = set(ALWAYS_DROP)
    if feature_set == "deployable":
        banned |= set(SATELLITE_ONLY_COLUMNS) | set(DEPLOYABLE_DROP)

    # A lag or rolling column inherits the eligibility of its base
    # column, so dropping 'aod' must also drop 'aod_lag_6'. Prefix
    # matching is safe because every derived name is '<base>_lag_<n>' or
    # '<base>_roll_<stat>_<n>'.
    def is_banned(column: str) -> bool:
        if column in banned:
            return True
        return any(
            column.startswith(f"{base}_lag_")
            or column.startswith(f"{base}_roll_")
            for base in banned
        )

    return [c for c in df.columns if not is_banned(c)]


def assert_no_leakage(columns) -> None:
    """Fail if an observed-past frame contains a known-future column.

    The two sides of the model are joined by :mod:`solarfc.dataset`, and
    the ``kf_`` prefix is the marker that a column was evaluated at the
    target timestamp rather than the origin. Finding one here means an
    assembly step ran in the wrong order.
    """
    offenders = [c for c in columns if str(c).startswith("kf_")]
    if offenders:
        raise AssertionError(
            f"observed-past frame contains known-future columns: {offenders}"
        )
