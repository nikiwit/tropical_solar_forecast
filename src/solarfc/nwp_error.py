"""Measure real NWP forecast error and fit the degradation model.

The realistic known-future track perturbs ERA5 by an amount that should
reflect how wrong a real weather forecast actually is. Published
verification figures are overwhelmingly European or CONUS, and tropical
convective cloud is a harder and different regime, so borrowing those
numbers would assert a calibration rather than establish one. This
module measures it instead, from archived JMA GSM forecasts over the
seven study sites.

What counts as "error"
----------------------
Error is measured as **JMA at lead L against JMA at lead 0**, not against ERA5.

Comparing JMA against ERA5 would conflate two different things: how much
a forecast degrades with lead time, and how much two different models
disagree at any lead. The second is inter-model spread and has nothing
to do with forecasting skill. Differencing a model against its own
shortest-lead run isolates the part we actually want.

ERA5 is still used, as an independent check that JMA's lead-0 run is a
reasonable stand-in for the analysis. That is reported, not fitted.

Saturation
----------
Forecast error grows with lead time but does not grow without bound: past a few
days a forecast is no better than climatology, so the error saturates at the
climatological spread of the variable. With leads only out to 72 h the
saturation point is not directly observable, so ``sigma_max`` is set from the
variable's own climatological standard deviation rather than extrapolated from
a short lever arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, SITE_KEYS, STEP_MINUTES
from .covariates import ErrorModel

__all__ = [
    "LEAD_HOURS",
    "load_jma",
    "load_era5_hourly",
    "error_series",
    "measure_error",
    "measure_error_autocorrelation",
    "fit_error_model",
    "fit_all_sites",
    "validate_fit",
]

#: Lead offsets present in the JMA archive. 72 h is unavailable for
#: 2018.
LEAD_HOURS: tuple[int, ...] = (0, 24, 48, 72)


def load_jma(site: str, data_dir=None) -> pd.DataFrame:
    """Load cached JMA forecasts for one site."""
    base = (PROCESSED_DIR / "jma") if data_dir is None else data_dir
    path = base / f"{site}_forecasts.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/pull_jma_forecasts.py first"
        )
    return pd.read_parquet(path)


@dataclass(frozen=True)
class ErrorStats:
    """Measured forecast error for one variable at one lead time."""

    variable: str
    lead_hours: int
    bias: float
    std: float
    rmse: float
    n: int


def load_era5_hourly(site: str, data_dir=None) -> pd.DataFrame:
    """ERA5 cache resampled back to its native hourly resolution.

    The cache is stored at 10-minute resolution for model input, but the
    intermediate values are interpolated. Comparing a forecast against
    them would measure interpolation artefacts alongside forecast error,
    so the comparison uses the hourly points ERA5 actually provides.
    """
    base = (PROCESSED_DIR / "era5") if data_dir is None else data_dir
    path = base / f"{site}_10min.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/build_era5_cache.py first"
        )
    return pd.read_parquet(path).resample("1h").first()


def measure_error(
    jma: pd.DataFrame,
    variable: str,
    lead_hours: int,
    *,
    mode: str = "total",
    era5: pd.DataFrame | None = None,
) -> ErrorStats | None:
    """Forecast error at a given lead time, under one of two definitions.

    Parameters
    ----------
    mode : {"total", "drift"}
        ``"total"`` compares the forecast against **ERA5**, which is the
        field the realistic track perturbs. This is what a deployed
        model actually experiences: it was trained on ERA5 and is fed a
        real forecast, so the input differs by forecast decay *plus*
        model bias *plus* resolution mismatch. None of those cancel, and
        all three degrade performance.

        ``"drift"`` compares the forecast against the same model's own
        lead-0 run. Model bias and resolution cancel, isolating pure
        forecast decay. Cleaner as a statement about NWP skill, but it
        understates what a plant experiences by 1.2-2.3x as measured
        over these sites, so it is reported rather than used to
        calibrate the degradation.

    Returns None when the required columns are absent or empty, so a
    partially-populated archive (72 h is unavailable for 2018) yields
    fewer fitting points rather than raising.
    """
    series = error_series(jma, variable, lead_hours, mode=mode, era5=era5)
    if series is None or series.empty:
        return None

    return ErrorStats(
        variable=variable,
        lead_hours=lead_hours,
        bias=float(series.mean()),
        std=float(series.std()),
        rmse=float(np.sqrt((series**2).mean())),
        n=int(len(series)),
    )


def error_series(
    jma: pd.DataFrame,
    variable: str,
    lead_hours: int,
    *,
    mode: str = "total",
    era5: pd.DataFrame | None = None,
) -> pd.Series | None:
    """The raw error series behind :func:`measure_error`.

    Exposed separately so several sites can be pooled by concatenating
    their error series. Pooling the frames instead would create
    duplicate timestamps and misalign the ERA5 join across sites.
    """
    if mode not in ("total", "drift"):
        raise ValueError(f"mode must be 'total' or 'drift', got {mode!r}")

    forecast = f"{variable}_lead{lead_hours}h"
    if forecast not in jma.columns:
        return None

    if mode == "drift":
        reference = f"{variable}_lead0h"
        if reference not in jma.columns:
            return None
        if lead_hours == 0:
            # Differenced against itself: identically zero. Kept so the
            # lead-0 row exists in the table, but it carries no growth
            # information and is excluded from the slope fit.
            n = int(jma[reference].notna().sum())
            return pd.Series(np.zeros(n), dtype=float) if n else None
        pair = jma[[reference, forecast]].dropna()
        return None if pair.empty else pair[forecast] - pair[reference]

    if era5 is None:
        raise ValueError("mode='total' requires the era5 frame")
    if variable not in era5.columns:
        return None
    joined = jma[[forecast]].join(era5[[variable]], how="inner").dropna()
    return None if joined.empty else joined[forecast] - joined[variable]


def measure_error_autocorrelation(
    jma: pd.DataFrame,
    variable: str,
    lead_hours: int = 24,
    max_lag_hours: int = 24,
) -> pd.Series:
    """Autocorrelation of the forecast error, by lag in hours.

    Real forecast errors persist -- a run that is too cloudy at noon is
    usually still too cloudy at 13:00. Treating the error as white noise
    would let a model average it away across the decoder window and
    would understate the damage a real forecast does.
    """
    reference = f"{variable}_lead0h"
    forecast = f"{variable}_lead{lead_hours}h"
    if reference not in jma.columns or forecast not in jma.columns:
        return pd.Series(dtype=float)

    error = (jma[forecast] - jma[reference]).dropna()
    if len(error) < max_lag_hours * 2:
        return pd.Series(dtype=float)

    return pd.Series(
        {
            lag: float(error.autocorr(lag))
            for lag in range(0, max_lag_hours + 1)
        },
        name=f"{variable}_lead{lead_hours}h",
    )


def _fit_correlation_hours(autocorr: pd.Series) -> float:
    """e-folding time of the error autocorrelation, in hours.

    Fitted by least squares on log-autocorrelation against lag, using
    only lags where the autocorrelation is still meaningfully positive
    -- beyond that the values are noise and would drag the fit.
    """
    if autocorr.empty:
        return 6.0

    usable = autocorr[(autocorr > 0.05) & (autocorr.index > 0)]
    if len(usable) < 3:
        return 6.0

    lags = usable.index.to_numpy(dtype=float)
    slope, _ = np.polyfit(lags, np.log(usable.to_numpy()), 1)
    if slope >= 0:
        return 24.0
    return float(min(-1.0 / slope, 48.0))


def fit_error_model(
    jma: pd.DataFrame,
    variable: str,
    *,
    mode: str = "total",
    era5: pd.DataFrame | None = None,
    climatological_std: float | None = None,
    bounds: tuple[float | None, float | None] = (None, None),
) -> tuple[ErrorModel, pd.DataFrame]:
    """Fit an :class:`ErrorModel` to measured error growth for one variable.

    Returns the model and the per-lead measurements it was fitted to, so
    the fit can be inspected and reported rather than trusted.
    """
    rows = []
    for lead in LEAD_HOURS:
        stats = measure_error(jma, variable, lead, mode=mode, era5=era5)
        if stats is not None:
            rows.append(stats)

    if len(rows) < 2:
        raise ValueError(
            f"{variable}: need at least 2 usable lead times to fit, got {len(rows)}"
        )

    measured = pd.DataFrame([vars(r) for r in rows])

    if climatological_std is None:
        reference = f"{variable}_lead0h"
        climatological_std = (
            float(jma[reference].std())
            if reference in jma
            else float(measured["std"].max())
        )

    model = _model_from_stats(
        variable,
        measured,
        climatological_std=climatological_std,
        correlation_hours=_fit_correlation_hours(
            measure_error_autocorrelation(jma, variable, lead_hours=24)
        ),
        bounds=bounds,
    )
    return model, measured


def _model_from_stats(
    variable: str,
    measured: pd.DataFrame,
    *,
    climatological_std: float,
    correlation_hours: float,
    bounds: tuple[float | None, float | None] = (None, None),
) -> ErrorModel:
    """Turn per-lead error statistics into a fitted :class:`ErrorModel`."""
    # Lead 0 in drift mode is identically zero, so it carries no growth
    # information and is excluded from the slope fit.
    fitting = measured[measured["lead_hours"] > 0]
    if fitting.empty:
        raise ValueError(f"{variable}: no lead times above zero to fit")

    leads = fitting["lead_hours"].to_numpy(float)
    if len(fitting) >= 2:
        slope, intercept = np.polyfit(leads, fitting["std"].to_numpy(float), 1)
        bias_growth, bias_0 = np.polyfit(
            leads, fitting["bias"].to_numpy(float), 1
        )
    else:
        row = fitting.iloc[0]
        slope = float(row["std"]) / float(row["lead_hours"])
        intercept = 0.0
        bias_0, bias_growth = float(row["bias"]), 0.0

    # A negative intercept is physically meaningless; clamp to a small
    # positive value so very short leads still carry some error.
    sigma_0 = float(max(intercept, 0.01 * max(fitting["std"].max(), 1e-6)))

    # Error saturates at climatological spread: a forecast no better
    # than climatology carries exactly that error and cannot do worse on
    # average.
    sigma_max = float(max(climatological_std, measured["std"].max()))

    return ErrorModel(
        variable=variable,
        sigma_0=sigma_0,
        growth_rate=float(max(slope, 0.0)),
        sigma_max=sigma_max,
        correlation_hours=correlation_hours,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        bias_0=float(bias_0),
        bias_growth=float(bias_growth),
    )


#: Physical bounds enforced after perturbation, per variable.
VARIABLE_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "era5_cloud_cover": (0.0, 1.0),
    "era5_relative_humidity": (0.0, 100.0),
    "era5_precip_mm_h": (0.0, None),
    "era5_temp_c": (None, None),
    "era5_dewpoint_c": (None, None),
}


def fit_all_sites(
    sites=SITE_KEYS,
    variables=None,
    data_dir=None,
    era5_dir=None,
    mode: str = "total",
) -> tuple[dict[str, ErrorModel], pd.DataFrame]:
    """Fit one error model per variable, pooling all sites.

    Pooling is deliberate. The degradation is applied uniformly across
    the benchmark, so fitting per site would bake site-specific forecast
    quality into what is meant to be a generic realism adjustment.
    Per-site measurements are still returned so an outlier site stays
    visible.

    Both modes are measured and returned; only ``mode`` is used for the
    fitted models. Drift is reported for Chapter 4 as a clean statement
    of NWP skill decay over tropical SEA; total is what calibrates the
    realistic track.
    """
    variables = list(VARIABLE_BOUNDS) if variables is None else list(variables)

    rows, jma_frames, era5_frames = [], [], []
    for site in sites:
        jma = load_jma(site, data_dir=data_dir)
        era5 = load_era5_hourly(site, data_dir=era5_dir)
        jma_frames.append(jma)
        era5_frames.append(era5)
        for variable in variables:
            for lead in LEAD_HOURS:
                for measurement_mode in ("total", "drift"):
                    stats = measure_error(
                        jma, variable, lead, mode=measurement_mode, era5=era5
                    )
                    if stats is not None:
                        rows.append(
                            vars(stats)
                            | {"site": site, "mode": measurement_mode}
                        )

    measurements = pd.DataFrame(rows)

    # Aggregate per-site statistics rather than concatenating error
    # series.
    #
    # Concatenating would add *between-site* variance to the fit --
    # sites with different systematic biases would inflate the pooled
    # standard deviation above anything a single site experiences. The
    # degradation is applied per-site, so the target is the typical
    # within-site error. Variances are averaged (not standard
    # deviations, which would understate it) and weighted by sample
    # count.
    pooled = {}
    for variable in variables:
        subset = measurements[
            (measurements["variable"] == variable)
            & (measurements["mode"] == mode)
        ]
        if subset.empty:
            raise ValueError(f"{variable}: no measurements in mode {mode!r}")

        stats_rows = []
        for lead, group in subset.groupby("lead_hours"):
            weights = group["n"].to_numpy(float)
            if weights.sum() == 0:
                continue
            mean_variance = float(
                np.average(group["std"].to_numpy(float) ** 2, weights=weights)
            )
            stats_rows.append(
                {
                    "variable": variable,
                    "lead_hours": int(lead),
                    "bias": float(
                        np.average(
                            group["bias"].to_numpy(float), weights=weights
                        )
                    ),
                    "std": float(np.sqrt(mean_variance)),
                    "rmse": float(
                        np.sqrt(
                            np.average(
                                group["rmse"].to_numpy(float) ** 2,
                                weights=weights,
                            )
                        )
                    ),
                    "n": int(weights.sum()),
                }
            )

        if len(stats_rows) < 2:
            raise ValueError(
                f"{variable}: need at least 2 usable lead times to fit, "
                f"got {len(stats_rows)}"
            )

        climatology = float(
            pd.concat(
                [
                    j[f"{variable}_lead0h"]
                    for j in jma_frames
                    if f"{variable}_lead0h" in j
                ],
                ignore_index=True,
            ).std()
        )
        autocorr = pd.concat(
            [
                measure_error_autocorrelation(j, variable, 24)
                for j in jma_frames
            ],
            axis=1,
        ).mean(axis=1)

        pooled[variable] = _model_from_stats(
            variable,
            pd.DataFrame(stats_rows),
            climatological_std=climatology,
            correlation_hours=_fit_correlation_hours(autocorr),
            bounds=VARIABLE_BOUNDS.get(variable, (None, None)),
        )

    return pooled, measurements


def validate_fit(
    model: ErrorModel, measured: pd.DataFrame, *, tolerance: float = 0.25
) -> pd.DataFrame:
    """Check the fitted model reproduces the error it was fitted to.

    A degradation model that does not match its own fitting data is
    worse than no degradation at all, because it produces numbers that
    look principled and are not. Returns a per-lead comparison with a
    pass flag; the caller decides what to do about failures rather than
    having an exception thrown at them.
    """
    rows = []
    for _, row in measured.iterrows():
        lead = float(row["lead_hours"])
        if lead == 0:
            continue
        predicted = float(model.sigma_at(lead))
        observed = float(row["std"])
        relative = (
            abs(predicted - observed) / observed if observed > 0 else np.nan
        )
        rows.append(
            {
                "variable": model.variable,
                "lead_hours": int(lead),
                "measured_std": observed,
                "model_sigma": predicted,
                "relative_error": relative,
                "within_tolerance": (
                    bool(relative <= tolerance)
                    if np.isfinite(relative)
                    else False
                ),
            }
        )
    return pd.DataFrame(rows)
