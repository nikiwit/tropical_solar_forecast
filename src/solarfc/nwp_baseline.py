"""The operational NWP baseline -- what a plant is competing against.

Every other reference in this project is statistical or machine-learned.
None of them is what a real operator is using today, which is a numerical
weather prediction from a national met agency. AEMO's accreditation test
makes the distinction concrete: a self-forecast is accepted only if it
beats the *incumbent operational forecast* on both MAE and RMSE. Without
this baseline the project cannot speak to adoption at all.

What can and cannot be built here
---------------------------------
JMA GSM carries no shortwave radiation at any date, and archived forecast
GHI begins in 2024 for every model while NSRDB ends in 2020. There is no
overlap, so a true operational GHI forecast cannot be scored on the test
year. What *is* available is archived forecast **cloud cover** back to
2018, which is a genuine forecast rather than a reanalysis.

So the baseline converts forecast cloud to irradiance through a clear-sky
transmittance model. The cloud input is real NWP output at a real lead
time; the conversion is ours. That makes this a **cloud-driven NWP
baseline**, not an operational GHI forecast, and it must be labelled that
way wherever it is reported.

Two forms, and why both
-----------------------
``cloud_driven_ghi`` applies a fitted Kasten-Czeplak transmittance curve
to forecast cloud alone. It is the physical conversion, and it is what
most solar papers mean by "the NWP baseline".

:class:`MosModel` is model output statistics -- a regression over every
field the forecast carries, fitted on training years. This is what an
operator actually runs, because raw NWP output is post-processed before
anyone dispatches on it. Measured here it roughly doubles the explained
variance of the cloud-only form, so reporting only the cloud-only version
would understate the incumbent and flatter everything compared against
it.

Both are reported. The gap between them is itself a result: it says what
statistical post-processing is worth over tropical Southeast Asia.

On the literature coefficients
------------------------------
Kasten and Czeplak (1980) give ``csi = 1 - 0.75 * cc**3.4`` from European
data, and it does not transfer. At full overcast it predicts a clear-sky
index of 0.25; the seven sites here measure 0.63. Scored on the test year
it produces a *negative* coefficient of determination at every site --
worse than predicting the training mean. Fitted coefficients come out
near ``a = 0.37, b = 0.72`` instead of ``0.75, 3.4``: a far flatter curve.

The literature constants are kept in :data:`KASTEN_CZEPLAK` so the
comparison can be reproduced rather than asserted, but the fitted form is
the one to report.

Why the relation is so weak
---------------------------
Even fitted, cloud cover explains little: R^2 runs 0.04-0.14 across the
sites. Total cloud fraction over a coarse grid cell is a poor proxy for
point transmittance in the tropics, where thin cirrus is common and the
cloud-cover distribution is compressed against its upper bound.

The weakness is not uniform, and the pattern is physically coherent: the
equatorial sites are the worst (KL, Penang, Kota Kinabalu at r ~ -0.22)
and the off-equator sites the best (Ho Chi Minh, Bangkok, Manila at
r ~ -0.36 to -0.42). That is the same latitude gradient found in JMA
forecast skill by an entirely separate route, and it is the reason an
operational forecast is a low bar near the equator specifically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CSI_CLIP_MAX, STEP_MINUTES

__all__ = [
    "KASTEN_CZEPLAK",
    "LEAD_HOURS",
    "MosModel",
    "cloud_driven_ghi",
    "fit_mos",
    "fit_transmittance",
    "lead_for_horizon",
    "mos_features",
    "transmittance",
    "upsample_forecast",
]

#: Kasten and Czeplak (1980) coefficients, ``csi = 1 - a * cc**b``.
#: Retained for comparison only -- see the module docstring.
KASTEN_CZEPLAK: tuple[float, float] = (0.75, 3.4)

#: Lead offsets the Previous Runs archive populates for JMA GSM.
#: ``0`` is the analysis-time value: not a forecast, and useful only as a
#: ceiling on what the cloud-to-irradiance conversion could ever achieve.
LEAD_HOURS: tuple[int, ...] = (0, 24, 48, 72)

#: Forecast fields the MOS regression may use, as column stems.
MOS_STEMS: tuple[str, ...] = (
    "era5_cloud_cover",
    "era5_temp_c",
    "era5_relative_humidity",
    "era5_dewpoint_c",
    "era5_precip_mm_h",
)

#: Floor on transmittance. Overcast tropical daytime irradiance is a
#: large fraction of clear-sky, never near zero, and an unbounded curve
#: can go negative outside its fitting range.
TRANSMITTANCE_FLOOR = 0.02


def transmittance(cloud_cover, a: float, b: float):
    """Clear-sky index implied by cloud fraction, ``1 - a * cc**b``.

    Parameters
    ----------
    cloud_cover : array-like
        Total cloud fraction in [0, 1]. Values outside are clipped
        rather than raising: a forecast field can carry small numerical
        excursions past its own bounds.
    a, b : float
        Curve coefficients. See :data:`KASTEN_CZEPLAK` for the published
        values and :func:`fit_transmittance` to obtain site-specific
        ones.

    Returns
    -------
    ndarray
        Clear-sky index, bounded to
        ``[TRANSMITTANCE_FLOOR, 1.0]``. NaN cloud propagates to NaN.
    """
    cc = np.asarray(cloud_cover, dtype=float)
    out = 1.0 - a * np.power(np.clip(cc, 0.0, 1.0), b)
    return np.where(
        np.isfinite(cc), np.clip(out, TRANSMITTANCE_FLOOR, 1.0), np.nan
    )


def fit_transmittance(
    cloud_cover, csi, *, initial: tuple[float, float] = KASTEN_CZEPLAK
) -> tuple[float, float]:
    """Least-squares fit of ``a`` and ``b`` to observed transmittance.

    Fit on training years only -- the curve is a fitted artefact, and
    fitting it on the evaluation period would leak the answer into the
    baseline it is supposed to provide.

    Parameters
    ----------
    cloud_cover, csi : array-like
        Paired forecast cloud fraction and observed clear-sky index.
        Non-finite pairs are dropped.
    initial : tuple of float
        Starting point for the optimiser.

    Returns
    -------
    tuple of float
        ``(a, b)``. Falls back to ``initial`` if fewer than 100 usable
        pairs are supplied, so a thin site degrades to the published
        curve rather than to a fit on noise.
    """
    from scipy.optimize import curve_fit

    cc = np.asarray(cloud_cover, dtype=float).ravel()
    y = np.asarray(csi, dtype=float).ravel()
    if cc.shape != y.shape:
        raise ValueError(f"shape mismatch: cloud {cc.shape} vs csi {y.shape}")

    ok = np.isfinite(cc) & np.isfinite(y)
    if ok.sum() < 100:
        return tuple(float(v) for v in initial)  # type: ignore[return-value]

    (a, b), _ = curve_fit(
        lambda c, a_, b_: transmittance(c, a_, b_),
        cc[ok],
        y[ok],
        p0=list(initial),
        bounds=([0.0, 0.3], [1.0, 8.0]),
        maxfev=20000,
    )
    return float(a), float(b)


def cloud_driven_ghi(cloud_cover, clearsky_ghi, a: float, b: float):
    """Forecast GHI as ``clearsky * transmittance(cloud)``.

    The clear-sky envelope supplies the deterministic solar geometry and
    the forecast supplies the atmosphere, which is the same decomposition
    smart persistence uses. Night is left to the envelope: where
    clear-sky is zero the product is zero regardless of cloud.
    """
    cs = np.asarray(clearsky_ghi, dtype=float).ravel()
    tau = transmittance(cloud_cover, a, b).ravel()
    if cs.shape != tau.shape:
        raise ValueError(f"shape mismatch: clearsky {cs.shape} vs {tau.shape}")
    return cs * tau


# --------------------------------------------------------------------------
# Model output statistics
# --------------------------------------------------------------------------


def mos_features(frame: pd.DataFrame, lead_hours: int) -> pd.DataFrame:
    """Assemble the MOS design matrix for one lead time.

    The forecast fields at that lead, plus calendar harmonics. Calendar
    terms are free -- they are known exactly for any future timestamp,
    so including them costs the baseline nothing in realism and stops
    the regression from having to express the diurnal and seasonal
    cycles through the weather fields.
    """
    columns = [
        f"{stem}_lead{lead_hours}h"
        for stem in MOS_STEMS
        if f"{stem}_lead{lead_hours}h" in frame.columns
    ]
    if not columns:
        raise ValueError(f"no forecast fields at lead {lead_hours}h")

    out = frame[columns].copy()
    index = pd.DatetimeIndex(frame.index)
    hour = index.hour + index.minute / 60.0
    doy = np.asarray(index.dayofyear, dtype=float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


@dataclass(frozen=True)
class MosModel:
    """A fitted ridge regression from forecast fields to clear-sky index.

    Deliberately a linear model solved in closed form rather than a
    gradient-boosted one. Three reasons: it is the incumbent's
    post-processing step, and operational MOS is overwhelmingly linear;
    it cannot quietly out-model the thing it is a baseline for; and it
    reduces to a dot product, so the C++ engine can reproduce it without
    a second inference path.

    Attributes
    ----------
    columns : tuple of str
        Design-matrix columns, in the order the coefficients assume.
    coef : ndarray
        Coefficients on standardised features.
    intercept : float
        Fitted intercept, in clear-sky-index units.
    mean, scale : ndarray
        Standardisation fitted on the training rows.
    """

    columns: tuple[str, ...]
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    lead_hours: int

    def predict_csi(self, frame: pd.DataFrame):
        """Predicted clear-sky index, clipped to the project's CSI range."""
        design = mos_features(frame, self.lead_hours)
        missing = [c for c in self.columns if c not in design.columns]
        if missing:
            raise ValueError(f"design matrix is missing {missing}")

        x = design[list(self.columns)].to_numpy(dtype=float)
        z = (x - self.mean) / self.scale
        out = z @ self.coef + self.intercept
        # A row with any missing forecast field cannot be predicted.
        out = np.where(np.isfinite(x).all(axis=1), out, np.nan)
        return np.clip(out, 0.0, CSI_CLIP_MAX)

    def predict_ghi(self, frame: pd.DataFrame, clearsky_ghi):
        """Predicted GHI, ``csi * clearsky``."""
        cs = np.asarray(clearsky_ghi, dtype=float).ravel()
        csi = np.asarray(self.predict_csi(frame), dtype=float).ravel()
        if cs.shape != csi.shape:
            raise ValueError(
                f"shape mismatch: clearsky {cs.shape} vs csi {csi.shape}"
            )
        return cs * csi


def fit_mos(
    frame: pd.DataFrame,
    csi,
    lead_hours: int,
    *,
    alpha: float = 1.0,
) -> MosModel:
    """Fit :class:`MosModel` by ridge regression on standardised features.

    Solved with ``numpy.linalg.lstsq`` on the augmented system rather
    than through scikit-learn, to keep the baseline free of a modelling
    dependency and trivially portable to the C++ engine.

    Parameters
    ----------
    frame : DataFrame
        Forecast fields, indexed by valid time. Restrict to training
        years before calling -- this function does not filter, and
        fitting on the evaluation period would invalidate the baseline.
    csi : array-like
        Observed clear-sky index on the same index.
    lead_hours : int
        Which forecast lead the design matrix is built from.
    alpha : float
        Ridge penalty on the standardised coefficients.
    """
    design = mos_features(frame, lead_hours)
    y = np.asarray(csi, dtype=float).ravel()
    if len(design) != y.size:
        raise ValueError(f"length mismatch: {len(design)} rows vs {y.size}")

    x = design.to_numpy(dtype=float)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if ok.sum() < 100:
        raise ValueError(f"only {int(ok.sum())} usable rows to fit MOS")

    x, y = x[ok], y[ok]
    mean = x.mean(axis=0)
    # A constant column would divide by zero; leave it unscaled.
    scale = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
    z = (x - mean) / scale

    # Ridge as an augmented least-squares problem: stacking sqrt(alpha)*I
    # below the design matrix penalises the coefficients without
    # penalising the intercept, which is handled by centring y.
    y_mean = float(y.mean())
    penalty = np.sqrt(alpha) * np.eye(z.shape[1])
    stacked = np.vstack([z, penalty])
    target = np.concatenate([y - y_mean, np.zeros(z.shape[1])])
    coef, *_ = np.linalg.lstsq(stacked, target, rcond=None)

    return MosModel(
        columns=tuple(design.columns),
        coef=coef,
        intercept=y_mean,
        mean=mean,
        scale=scale,
        lead_hours=lead_hours,
    )


# --------------------------------------------------------------------------
# Aligning an hourly forecast onto the evaluation grid
# --------------------------------------------------------------------------


def lead_for_horizon(
    horizon_steps: int,
    *,
    step_minutes: int = STEP_MINUTES,
    leads: tuple[int, ...] = LEAD_HOURS,
) -> int:
    """Smallest archived lead that a forecast at this horizon could use.

    A forecast valid ``h`` hours ahead has to come from a run issued at
    least ``h`` hours earlier, so the operationally honest choice is the
    shortest available lead that is not shorter than the horizon.

    Lead 0 is excluded from the result because it is an analysis rather
    than a forecast. Sub-daily horizons therefore map to the 24-hour
    lead, which understates what a fresh run would provide -- the
    archive serves fixed daily offsets and has nothing between 0 and 24
    hours. That penalty is real and belongs in the write-up; it is also
    why the 24-hour and 48-hour horizons are where this comparison
    carries the most weight, since there the mapping is exact.
    """
    horizon_hours = horizon_steps * step_minutes / 60.0
    usable = [lead for lead in leads if lead > 0 and lead >= horizon_hours]
    if not usable:
        return max(leads)
    return min(usable)


def upsample_forecast(
    frame: pd.DataFrame, target: pd.DatetimeIndex
) -> pd.DataFrame:
    """Reindex an hourly forecast onto the evaluation grid.

    Linear interpolation for every field, matching
    :func:`solarfc.era5.upsample_to_10min`. The forecast stores point
    samples of a smooth field, so linear is the correct reconstruction
    *of the stored series*. Neither this nor a forward fill recovers
    genuine sub-hourly cloud dynamics, which is a limitation of the
    input rather than of the fill rule.
    """
    target = pd.DatetimeIndex(target)
    union = frame.index.union(target)
    return (
        frame.reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(target)
    )
