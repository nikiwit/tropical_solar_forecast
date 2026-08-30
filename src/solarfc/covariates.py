"""Known-future covariates for the decoder, in three tracks.

Why three tracks
----------------
The original design gave the decoder only calendar and astronomical inputs.
That left it atmospherically empty: beyond roughly 3-6 hours the model had no
information source except climatology shaped by the diurnal cycle, so Forecast
Skill at the long horizons would collapse and the decoder half of the TFT --
the specific reason a TFT was chosen over a plain encoder -- would go unused.

Feeding unmodified ERA5 over the forecast window fixes that but overshoots in
the other direction. ERA5 is reanalysis, so it is a *perfect* weather forecast.
No operator will ever have one, and reporting those numbers as operational
accuracy is the most common criticism levelled at solar forecasting papers.

So every model trains and reports on three tracks:

===================  ==========================================  ==================
Track                Known-future atmospheric inputs             What it measures
===================  ==========================================  ==================
``nwp_free``         none                                        honest lower bound
``realistic``        ERA5 degraded by measured forecast error    the operational number
``perfect``          ERA5 unmodified                             optimistic ceiling
===================  ==========================================  ==================

The middle one is what a plant would actually achieve, and it is the one most
papers omit.

Decoder window
--------------
One decoder sequence spans the maximum horizon, and every horizon is read off
that single pass. This is how TFT works in Lim et al. (2021) and is the natural
design for a MIMO head. It means a 20-minute forecast can attend to weather
48 hours out, which is operationally odd but not leakage -- NWP fields for the
whole window genuinely are available at issue time, which is the entire point
of a known-future input.

Leakage
-------
Known-future covariates are the one place in this pipeline where future
information is legitimate, which makes it the one place where genuine leakage
would be hardest to notice. Two rules, both asserted in tests rather than by
inspection:

1. Only *deterministic* or *forecast* quantities may appear. Solar geometry and
   calendar terms are computable years ahead. NWP fields are forecasts. Observed
   irradiance is neither and must never appear in the decoder.
2. The ``nwp_free`` track must contain no atmospheric column at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    HORIZON_STEPS,
    MONTH_TO_MONSOON,
    STEP_MINUTES,
    SITES_BY_KEY,
    Site,
)

__all__ = [
    "TRACKS",
    "CALENDAR_FEATURES",
    "SOLAR_FEATURES",
    "NWP_FEATURES",
    "solar_geometry",
    "calendar_features",
    "build_known_future",
    "known_future_columns",
    "load_fitted_error_models",
]

#: The three known-future regimes. Order is lower bound, operational, ceiling.
TRACKS: tuple[str, ...] = ("nwp_free", "realistic", "perfect")

#: Deterministic from the timestamp alone.
CALENDAR_FEATURES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "monsoon_phase",
)

#: Deterministic from timestamp plus site coordinates. Computable years ahead,
#: which is what makes them legitimate known-future inputs.
SOLAR_FEATURES: tuple[str, ...] = (
    "solar_zenith",
    "solar_azimuth",
    "clearsky_ghi",
    "cos_zenith",
)

#: Atmospheric fields. Present only in the realistic and perfect tracks.
NWP_FEATURES: tuple[str, ...] = (
    "era5_cloud_cover",
    "era5_temp_c",
    "era5_dewpoint_c",
    "era5_relative_humidity",
    "era5_precip_mm_h",
)


def known_future_columns(track: str) -> tuple[str, ...]:
    """Columns the decoder receives for a given track."""
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    base = CALENDAR_FEATURES + SOLAR_FEATURES
    return base if track == "nwp_free" else base + NWP_FEATURES


# --------------------------------------------------------------------------
# Deterministic covariates
# --------------------------------------------------------------------------


def solar_geometry(index: pd.DatetimeIndex, site: Site | str) -> pd.DataFrame:
    """Solar position and clear-sky irradiance from timestamp and coordinates.

    Uses pvlib: SPA for solar position, Ineichen with the Linke turbidity
    climatology for clear-sky GHI.

    These are the backbone of the known-future path. They carry no weather
    information at all, but they encode the hard physical constraint that
    irradiance is zero at night and bounded by the clear-sky envelope by day --
    which is why even the NWP-free track is not vacuous.
    """
    import pvlib

    if isinstance(site, str):
        site = SITES_BY_KEY[site]

    location = pvlib.location.Location(
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.elevation,
        tz="UTC",
    )

    position = location.get_solarposition(index)
    clearsky = location.get_clearsky(index, model="ineichen")

    out = pd.DataFrame(index=index)
    out["solar_zenith"] = position["zenith"].to_numpy()
    out["solar_azimuth"] = position["azimuth"].to_numpy()
    out["clearsky_ghi"] = clearsky["ghi"].to_numpy()
    # Cosine of zenith is the geometric driver of irradiance and is better
    # behaved for a network than the angle: it varies smoothly through solar
    # noon and goes negative at night rather than saturating at 90 degrees.
    out["cos_zenith"] = np.cos(np.radians(out["solar_zenith"]))
    return out


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical time encodings and monsoon phase.

    Hour and day-of-year are encoded as sin/cos pairs so that 23:50 and 00:00
    are adjacent rather than maximally distant, which a raw integer would imply.
    """
    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=idx)

    minute_of_day = idx.hour * 60 + idx.minute
    angle = 2.0 * np.pi * minute_of_day / (24 * 60)
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)

    day_angle = 2.0 * np.pi * np.asarray(idx.dayofyear, dtype=float) / 365.25
    out["doy_sin"] = np.sin(day_angle)
    out["doy_cos"] = np.cos(day_angle)

    months = np.asarray(idx.month, dtype=int)
    phase = np.empty(months.shape, dtype=np.int8)
    for month, code in MONTH_TO_MONSOON.items():
        phase[months == month] = code
    out["monsoon_phase"] = phase

    return out


# --------------------------------------------------------------------------
# Forecast-error degradation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorModel:
    """Parametric NWP forecast-error model for one variable.

    Fitted to error measured from JMA GSM archived forecasts (Open-Meteo
    Previous Runs) rather than taken from published verification figures, which
    are predominantly European or CONUS. Tropical convective cloud is a harder
    and different regime, so borrowing those numbers would assert a calibration
    rather than establish one.

    Error grows with lead time as ``sigma_0 + growth_rate * lead_hours``, capped
    at ``sigma_max`` because forecast error saturates at climatological spread
    rather than growing without bound.

    ``correlation_hours`` sets the timescale over which the error is smooth.
    Real forecast errors are persistent -- a run that is too cloudy at noon is
    usually still too cloudy at 13:00 -- so white noise would understate the
    damage by letting a model average the error away across the window.
    """

    variable: str
    sigma_0: float
    growth_rate: float
    sigma_max: float
    correlation_hours: float = 6.0
    lower_bound: float | None = None
    upper_bound: float | None = None
    #: Systematic offset between the forecast and the field being perturbed.
    #: Measured against ERA5, most of the total error is a lead-independent
    #: offset from model bias and resolution mismatch rather than forecast
    #: decay. A bias is a shift, not zero-mean noise, so applying it as noise
    #: would understate how wrong the input actually is.
    bias_0: float = 0.0
    bias_growth: float = 0.0

    def sigma_at(self, lead_hours):
        """Error standard deviation at each lead time, saturating at sigma_max."""
        lead = np.asarray(lead_hours, dtype=float)
        return np.minimum(self.sigma_0 + self.growth_rate * lead, self.sigma_max)

    def bias_at(self, lead_hours):
        """Systematic offset at each lead time."""
        lead = np.asarray(lead_hours, dtype=float)
        return self.bias_0 + self.bias_growth * lead


def _correlated_noise(n: int, correlation_steps: float, rng) -> np.ndarray:
    """Unit-variance noise smoothed to a given correlation length.

    An AR(1)-style exponential smoothing, renormalised so the output has unit
    variance regardless of the correlation length -- otherwise a longer
    correlation would silently shrink the error magnitude.
    """
    if n <= 0:
        return np.zeros(0, dtype=float)
    if correlation_steps <= 1.0:
        return rng.standard_normal(n)

    phi = np.exp(-1.0 / correlation_steps)
    white = rng.standard_normal(n)
    out = np.empty(n, dtype=float)
    out[0] = white[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + np.sqrt(1.0 - phi**2) * white[i]
    return out


def degrade(
    values,
    model: ErrorModel,
    lead_hours,
    *,
    rng=None,
) -> np.ndarray:
    """Apply lead-time-dependent, temporally correlated error to a forecast field.

    Physical bounds are enforced after perturbation: cloud cover cannot leave
    [0, 1] and precipitation cannot go negative, no matter what the noise does.
    Clipping is applied last so the bounds hold exactly rather than in
    distribution.
    """
    rng = np.random.default_rng() if rng is None else rng
    v = np.asarray(values, dtype=float).ravel()
    sigma = np.broadcast_to(np.asarray(model.sigma_at(lead_hours), dtype=float), v.shape)
    bias = np.broadcast_to(np.asarray(model.bias_at(lead_hours), dtype=float), v.shape)

    correlation_steps = model.correlation_hours * 60.0 / STEP_MINUTES
    noise = _correlated_noise(v.size, correlation_steps, rng)

    out = v + bias + sigma * noise
    if model.lower_bound is not None:
        out = np.maximum(out, model.lower_bound)
    if model.upper_bound is not None:
        out = np.minimum(out, model.upper_bound)
    return out


def load_fitted_error_models(path=None) -> dict[str, ErrorModel]:
    """Load error models fitted to measured JMA GSM forecast error.

    Produced by ``scripts/pull_jma_forecasts.py`` followed by
    ``solarfc.nwp_error.fit_all_sites``. Falls back to the provisional
    parameters when the fit has not been run, so the pipeline still works on a
    fresh checkout -- but anything reported must use the fitted values.
    """
    import json

    from .config import PROCESSED_DIR

    path = (PROCESSED_DIR / "nwp_error" / "error_models.json") if path is None else path
    if not path.exists():
        return dict(PROVISIONAL_ERROR_MODELS)

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: ErrorModel(**params) for name, params in raw.items()}


#: Placeholder parameters, used only when the fitted models are unavailable.
#:
#: These are deliberately NOT for any reported result. The fitted values come
#: from measured JMA GSM error over the seven study sites -- see
#: ``solarfc.nwp_error``. Using these for a headline number would be exactly
#: the asserted-not-measured calibration this design exists to avoid.
PROVISIONAL_ERROR_MODELS: dict[str, ErrorModel] = {
    "era5_cloud_cover": ErrorModel(
        "era5_cloud_cover", 0.10, 0.004, 0.35, 6.0, 0.0, 1.0
    ),
    "era5_temp_c": ErrorModel("era5_temp_c", 0.8, 0.03, 3.0, 12.0),
    "era5_dewpoint_c": ErrorModel("era5_dewpoint_c", 1.0, 0.035, 3.5, 12.0),
    "era5_relative_humidity": ErrorModel(
        "era5_relative_humidity", 4.0, 0.15, 15.0, 12.0, 0.0, 100.0
    ),
    "era5_precip_mm_h": ErrorModel(
        "era5_precip_mm_h", 0.3, 0.02, 1.5, 3.0, 0.0, None
    ),
}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_known_future(
    index: pd.DatetimeIndex,
    site: Site | str,
    track: str = "nwp_free",
    era5: pd.DataFrame | None = None,
    *,
    error_models: dict[str, ErrorModel] | None = None,
    lead_hours=None,
    rng=None,
) -> pd.DataFrame:
    """Assemble the decoder's known-future covariates for one track.

    Parameters
    ----------
    index : DatetimeIndex
        Timestamps of the forecast window, UTC.
    site : Site or str
        Determines solar geometry.
    track : {"nwp_free", "realistic", "perfect"}
    era5 : DataFrame, optional
        Upsampled ERA5 features. Required for the realistic and perfect tracks.
    error_models : dict, optional
        Per-variable error models for the realistic track. Defaults to the
        provisional set, which must be replaced by JMA-fitted values before any
        reported result.
    lead_hours : array-like, optional
        Lead time of each row. Required for the realistic track, since the
        degradation is lead-dependent.

    Returns
    -------
    DataFrame
        Indexed by ``index``, with exactly the columns
        ``known_future_columns(track)`` in that order.
    """
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")

    idx = pd.DatetimeIndex(index)
    out = pd.concat([calendar_features(idx), solar_geometry(idx, site)], axis=1)

    if track == "nwp_free":
        return out[list(known_future_columns(track))]

    if era5 is None:
        raise ValueError(f"track {track!r} requires ERA5 features, got era5=None")

    missing = [c for c in NWP_FEATURES if c not in era5.columns]
    if missing:
        raise ValueError(f"ERA5 frame is missing required columns: {missing}")

    nwp = era5.reindex(idx)[list(NWP_FEATURES)]

    if track == "realistic":
        if lead_hours is None:
            raise ValueError(
                "track 'realistic' requires lead_hours -- the degradation is "
                "lead-time dependent and applying it without lead times would "
                "silently use a single error magnitude for every horizon"
            )
        models = load_fitted_error_models() if error_models is None else error_models
        rng = np.random.default_rng() if rng is None else rng
        nwp = nwp.copy()
        for column in NWP_FEATURES:
            if column in models:
                nwp[column] = degrade(
                    nwp[column].to_numpy(), models[column], lead_hours, rng=rng
                )

    out = pd.concat([out, nwp], axis=1)
    return out[list(known_future_columns(track))]


def lead_hours_for_window(
    origin: pd.Timestamp, index: pd.DatetimeIndex
) -> np.ndarray:
    """Lead time in hours of each timestamp relative to the forecast origin.

    Negative values mean the timestamp precedes the origin, which should never
    happen in a decoder window and is left unclipped so a caller's mistake
    surfaces rather than being silently absorbed.
    """
    delta = pd.DatetimeIndex(index) - pd.Timestamp(origin)
    return delta.total_seconds().to_numpy() / 3600.0


#: Maximum decoder span, in steps. One pass covers this and every horizon is
#: read from it -- standard TFT, and the natural design for a MIMO head.
MAX_DECODER_STEPS: int = max(HORIZON_STEPS)
