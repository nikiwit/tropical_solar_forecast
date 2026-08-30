"""ERA5 reanalysis loading, site extraction and upsampling to the NSRDB grid.

ERA5 provides the NWP-side features: cloud cover, temperature, humidity,
wind, precipitation. It is hourly at 0.25 degrees; NSRDB is 10-minute at
2 km. This module bridges that gap.

Three things here are easy to get wrong, and all three are handled
explicitly.

1. The files are ZIP archives, not NetCDF
--------------------------------------------------------------------------
The Copernicus CDS API delivers ``.nc`` files that are actually zip containers
holding two NetCDF members — one for instantaneous fields and one for
accumulated fields. Opening them directly with xarray fails. :func:`open_month`
handles the container transparently.

2. Accumulated fields are backward-looking
--------------------------------------------------------------------------
``tp`` and ``ssrd`` are accumulated over the hour **ending** at ``valid_time``.
The value stamped 07:00 describes 06:00-07:00. Forward-filling it onto
06:00..06:50 is correct; forward-filling onto 07:00..07:50 — which is what a
naive resample does — misattributes every accumulation by a full hour and
would put solar radiation in the wrong part of the day. See
:func:`upsample_to_10min`.

3. Instantaneous and accumulated fields need different upsampling
--------------------------------------------------------------------------
Instantaneous fields are point samples of a continuous field, so linear
interpolation between them is the correct reconstruction. Accumulations are
integrals over a window: they are converted to a mean rate and held constant
across the window they describe. Interpolating an accumulation would smear
energy across hour boundaries and is simply wrong.

Known limitation, to be stated in the thesis
--------------------------------------------------------------------------
Interpolating hourly cloud cover to 10 minutes cannot create real sub-hourly
cloud dynamics. The interpolated series is smooth where reality is abrupt,
which is precisely the regime the 20-minute and 30-minute horizons operate in.
This is an acknowledged limitation of using reanalysis rather than satellite
cloud motion, and it is why NSRDB (natively 10-minute, satellite-derived)
carries the fast-varying signal while ERA5 supplies the slow synoptic context.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import ERA5_DIR, SITES_BY_KEY, STEP_MINUTES, Site

__all__ = [
    "INSTANT_VARS",
    "ACCUM_VARS",
    "open_month",
    "extract_site",
    "load_site_era5",
    "upsample_to_10min",
    "derive_features",
]

#: Point-in-time fields. Linear interpolation is the correct
#: reconstruction.
INSTANT_VARS: tuple[str, ...] = (
    "tcc",
    "t2m",
    "d2m",
    "msl",
    "u10",
    "v10",
    "tcwv",
)

#: Fields accumulated over the hour ENDING at valid_time. Converted to a
#: mean rate and held constant across the window they describe.
ACCUM_VARS: tuple[str, ...] = ("tp", "ssrd")

_ZIP_MEMBERS = (
    "data_stream-oper_stepType-instant.nc",
    "data_stream-oper_stepType-accum.nc",
)


def open_month(path: str | Path) -> xr.Dataset:
    """Open one monthly ERA5 file, transparently handling the zip container.

    Returns a single dataset with both instantaneous and accumulated
    variables merged on ``valid_time``. Plain NetCDF files are also
    accepted, so this keeps working if CDS reverts its delivery format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {path}")

    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path)

    parts = []
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith(".nc")]
        if not members:
            raise ValueError(f"{path.name} is a zip with no NetCDF members")
        for name in members:
            # Read into memory: these members are ~100 MB and xarray
            # cannot lazily seek inside a zip anyway.
            parts.append(xr.open_dataset(z.read(name)))

    merged = xr.merge(parts, compat="override", join="exact")

    # Singleton coordinates that only complicate downstream selection.
    for coord in ("number", "expver"):
        if coord in merged.coords and merged.coords[coord].size == 1:
            merged = merged.drop_vars(coord)

    return merged


def extract_site(ds: xr.Dataset, site: Site | str) -> pd.DataFrame:
    """Nearest-gridpoint time series for one site, as a UTC-indexed frame.

    ERA5's 0.25-degree grid puts the nearest point up to ~20 km from the
    target coordinate. That offset is recorded in the frame's ``attrs``
    so the methodology chapter can quote it rather than estimate it.
    """
    if isinstance(site, str):
        site = SITES_BY_KEY[site]

    point = ds.sel(
        latitude=site.latitude, longitude=site.longitude, method="nearest"
    )

    available = [
        v for v in (*INSTANT_VARS, *ACCUM_VARS) if v in point.data_vars
    ]
    df = point[available].to_dataframe()

    # to_dataframe carries the scalar lat/lon coords through as columns.
    df = df.drop(
        columns=[c for c in ("latitude", "longitude") if c in df.columns]
    )

    if "valid_time" in df.index.names:
        df.index = pd.DatetimeIndex(
            df.index.get_level_values("valid_time"), tz="UTC"
        )
    else:
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
    df.index.name = "timestamp"

    grid_lat = float(point.latitude)
    grid_lon = float(point.longitude)
    df.attrs.update(
        site=site.key,
        site_lat=site.latitude,
        site_lon=site.longitude,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        offset_km=_haversine_km(
            site.latitude, site.longitude, grid_lat, grid_lon
        ),
    )
    return df.sort_index()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, for reporting the gridpoint offset."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def load_site_era5(
    site: Site | str, years, months=range(1, 13), data_dir=None
):
    """Load and concatenate monthly ERA5 files for one site.

    Reads each month, extracts the nearest gridpoint, then discards the
    full grid — holding 60 months of 97x97x744 in memory is unnecessary
    when only seven points are ever used.
    """
    base = ERA5_DIR if data_dir is None else Path(data_dir)
    key = site if isinstance(site, str) else site.key

    frames, attrs = [], {}
    for year in years:
        for month in months:
            path = base / f"era5_sea_{year}_{month:02d}.nc"
            if not path.exists():
                raise FileNotFoundError(f"missing ERA5 month: {path}")
            with open_month(path) as ds:
                part = extract_site(ds, key)
            attrs = part.attrs
            frames.append(part)

    df = pd.concat(frames).sort_index()

    dupes = int(df.index.duplicated().sum())
    if dupes:
        raise ValueError(f"{key}: {dupes} duplicate ERA5 timestamps")

    df.attrs.update(attrs)
    return df


def upsample_to_10min(
    df: pd.DataFrame,
    *,
    step_minutes: int = STEP_MINUTES,
    cloud_method: str = "linear",
) -> pd.DataFrame:
    """Upsample hourly ERA5 to the 10-minute NSRDB grid.

    Parameters
    ----------
    df : DataFrame
        Hourly, UTC-indexed, as returned by :func:`load_site_era5`.
    step_minutes : int
        Target resolution.
    cloud_method : {"linear", "ffill"}
        How to upsample total cloud cover.

        ``"linear"`` (default) treats ``tcc`` as what ERA5 actually
        stores — an instantaneous field — and interpolates between
        samples. ``"ffill"`` holds each hourly value constant, which
        avoids inventing intermediate values but introduces a step
        discontinuity on the hour that a model can learn to key on.

        The plan originally specified forward-fill on the grounds that
        cloud cover is stepwise. That reasoning describes real cloud,
        not the ERA5 field, which is a point sample of a smooth
        analysis. Linear is the correct reconstruction *of the stored
        field*; neither option recovers genuine sub-hourly cloud
        dynamics, and that limitation belongs in the thesis rather than
        in a choice of fill rule.

    Returns
    -------
    DataFrame
        Reindexed to ``step_minutes``. Accumulated variables are
        returned as mean rates over their window, renamed with a
        ``_rate`` suffix to make the unit change impossible to miss
        downstream.
    """
    if cloud_method not in ("linear", "ffill"):
        raise ValueError(
            f"cloud_method must be 'linear' or 'ffill', got {cloud_method!r}"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"expected a DatetimeIndex, got {type(df.index).__name__}"
        )

    target = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq=f"{step_minutes}min",
        tz=df.index.tz,
    )
    out = pd.DataFrame(index=target)
    out.index.name = "timestamp"

    # --- Instantaneous fields: linear interpolation between point
    # samples ----
    for var in INSTANT_VARS:
        if var not in df.columns:
            continue
        if var == "tcc" and cloud_method == "ffill":
            out[var] = df[var].reindex(target, method="ffill")
        else:
            out[var] = (
                df[var]
                .reindex(df.index.union(target))
                .interpolate(method="time")
                .reindex(target)
            )

    # --- Accumulated fields: shift back, convert to rate, hold constant
    # ------
    #
    # The value stamped at T covers (T-1h, T]. Shifting the series back
    # by one hour re-stamps it onto the START of the window it
    # describes, after which a forward fill assigns it to exactly the
    # sub-steps it actually covers.
    seconds_per_hour = 3600.0
    for var in ACCUM_VARS:
        if var not in df.columns:
            continue
        window_start = df[var].copy()
        window_start.index = window_start.index - pd.Timedelta(hours=1)
        rate = window_start / seconds_per_hour
        out[f"{var}_rate"] = rate.reindex(target, method="ffill")

    out.attrs.update(df.attrs)
    out.attrs["cloud_method"] = cloud_method
    return out


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 units and derive the model-facing feature set.

    ERA5 ships SI units that do not match NSRDB's conventions, and it
    provides no relative humidity field at all. Aligning units here
    means the feature pipeline never has to care which dataset a column
    came from.
    """
    out = df.copy()

    if "t2m" in out:
        out["era5_temp_c"] = out["t2m"] - 273.15
    if "d2m" in out:
        out["era5_dewpoint_c"] = out["d2m"] - 273.15
    if "msl" in out:
        out["era5_pressure_hpa"] = out["msl"] / 100.0
    if "tcc" in out:
        out["era5_cloud_cover"] = out["tcc"]
    if "tcwv" in out:
        out["era5_precipitable_water_mm"] = out["tcwv"]  # kg/m^2 == mm

    if {"u10", "v10"}.issubset(out.columns):
        out["era5_wind_speed"] = np.hypot(out["u10"], out["v10"])
        # Meteorological convention: the direction the wind blows FROM.
        out["era5_wind_direction"] = (
            np.degrees(np.arctan2(-out["u10"], -out["v10"])) % 360.0
        )

    if "tp_rate" in out:
        # m/s -> mm/h. This is the rainfall proxy that conditions the
        # Monsoon Gate (Module A), so its units need to be unambiguous.
        out["era5_precip_mm_h"] = out["tp_rate"] * 1000.0 * 3600.0
    if "ssrd_rate" in out:
        out["era5_ssrd_wm2"] = out["ssrd_rate"]

    if {"era5_temp_c", "era5_dewpoint_c"}.issubset(out.columns):
        out["era5_relative_humidity"] = _relative_humidity(
            out["era5_temp_c"], out["era5_dewpoint_c"]
        )

    return out


def _relative_humidity(temp_c, dewpoint_c):
    """RH (%) from temperature and dewpoint via the Magnus-Tetens formula.

    ERA5 has no RH field, but humidity is a direct input to the Monsoon
    Gate and a well-established predictor of tropical diffuse fraction,
    so it is derived rather than dropped. Coefficients are the standard
    Magnus values over water, valid across the temperature range of
    every site here.
    """
    a, b = 17.625, 243.04
    gamma_d = (a * dewpoint_c) / (b + dewpoint_c)
    gamma_t = (a * temp_c) / (b + temp_c)
    return (100.0 * np.exp(gamma_d - gamma_t)).clip(0.0, 100.0)
