"""Deterministic and probabilistic forecast metrics.

Locked before any model is trained, so that every baseline and every
ablation variant is scored by identical code.

Conventions
-----------
* All functions take 1-D arrays of equal length and return floats.
* NaNs in either array are dropped pairwise. An all-NaN input returns NaN
  rather than raising, so a single empty stratum cannot abort a results
  sweep.
* Daytime masking is the caller's responsibility via :func:`daytime_mask`.
  Metrics do not silently filter, because a metric that quietly changes
  its own sample set is not comparable across models.
"""

from __future__ import annotations

import numpy as np

from .config import DAYTIME_CLEARSKY_FLOOR, MAPE_GHI_FLOOR

__all__ = [
    "daytime_mask",
    "mae",
    "rmse",
    "mbe",
    "mape",
    "nmae",
    "nrmse",
    "r2",
    "forecast_skill",
    "pinball_loss",
    "picp",
    "pinaw",
    "reliability_curve",
    "point_metrics",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _clean(y_true, y_pred):
    """Coerce to float arrays and drop pairwise-NaN samples."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok]


def daytime_mask(
    clearsky_ghi, floor: float = DAYTIME_CLEARSKY_FLOOR
) -> np.ndarray:
    """Boolean mask selecting daytime samples.

    Night-time GHI is identically zero and trivially predictable.
    Including it shrinks MAE, inflates R^2 toward 1, and makes MAPE
    undefined. Every headline metric in this project is computed on
    daytime samples only.

    Uses clear-sky GHI rather than measured GHI so the mask is a
    property of solar geometry alone and is therefore identical for
    every model compared.
    """
    cs = np.asarray(clearsky_ghi, dtype=float).ravel()
    return np.isfinite(cs) & (cs > floor)


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in the units of the input (W/m^2)."""
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (W/m^2)."""
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if t.size else float("nan")


def mbe(y_true, y_pred) -> float:
    """Mean bias error (W/m^2). Positive => the model over-predicts.

    Reported per monsoon phase to expose systematic seasonal bias, which
    a symmetric metric such as RMSE cannot reveal.
    """
    t, p = _clean(y_true, y_pred)
    return float(np.mean(p - t)) if t.size else float("nan")


def mape(y_true, y_pred, floor: float = MAPE_GHI_FLOOR) -> float:
    """Mean absolute percentage error (%), over samples with y_true > floor.

    MAPE diverges as the denominator approaches zero, which for
    irradiance happens twice daily. The floor makes it computable, but
    it also makes the sample set differ from the other metrics — so
    prefer :func:`nmae` and :func:`nrmse` when reporting, and treat MAPE
    as a familiarity metric only.
    """
    t, p = _clean(y_true, y_pred)
    ok = t > floor
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs((t[ok] - p[ok]) / t[ok])) * 100.0)


def nmae(y_true, y_pred) -> float:
    """MAE normalised by the mean of observations (%). Comparable across sites."""
    t, p = _clean(y_true, y_pred)
    if not t.size:
        return float("nan")
    denom = np.mean(t)
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(t - p)) / denom * 100.0)


def nrmse(y_true, y_pred) -> float:
    """RMSE normalised by the mean of observations (%)."""
    t, p = _clean(y_true, y_pred)
    if not t.size:
        return float("nan")
    denom = np.mean(t)
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return float("nan")
    return float(np.sqrt(np.mean((t - p) ** 2)) / denom * 100.0)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination.

    Note that R^2 on daytime irradiance is dominated by the diurnal
    cycle and will look high for almost any model. It is reported for
    convention, not as evidence of skill — Forecast Skill is the metric
    that matters.
    """
    t, p = _clean(y_true, y_pred)
    if t.size < 2:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def forecast_skill(y_true, y_pred, y_reference) -> float:
    """FS = 1 - RMSE_model / RMSE_reference.

    1.0 is a perfect forecast, 0.0 matches the reference, negative is
    worse than the reference.

    The reference must be **smart persistence**, not naive persistence,
    for the headline number. Naive persistence is a weak reference in a
    strongly diurnal signal and inflates FS for every model; see
    :func:`solarfc.baselines.smart_persistence`.
    """
    t, p = _clean(y_true, y_pred)
    t_ref, ref = _clean(y_true, y_reference)
    if not t.size or not t_ref.size:
        return float("nan")
    denom = np.sqrt(np.mean((t_ref - ref) ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sqrt(np.mean((t - p) ** 2)) / denom)


# --------------------------------------------------------------------------
# Probabilistic metrics
# --------------------------------------------------------------------------


def pinball_loss(y_true, y_pred_q, quantile: float) -> float:
    """Pinball (quantile) loss for a single quantile level.

    This is also the training objective for the MIMO quantile head.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    t, p = _clean(y_true, y_pred_q)
    if not t.size:
        return float("nan")
    diff = t - p
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def picp(y_true, lower, upper) -> float:
    """Prediction interval coverage probability, as a fraction in [0, 1].

    For a nominal P10/P90 interval a calibrated model returns ~0.80.
    Report this separately for monsoon-transition and stable windows:
    calibration characteristically degrades under regime shift, and an
    aggregate figure hides precisely that.
    """
    t = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if not (t.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must share a shape")
    ok = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    return float(np.mean((t[ok] >= lo[ok]) & (t[ok] <= hi[ok])))


def pinaw(y_true, lower, upper) -> float:
    """Prediction interval normalised average width.

    PICP alone is trivially gamed by predicting an enormous interval.
    PINAW is the cost side of that trade and must always be reported
    alongside it. Normalised by the observed range so it is comparable
    across sites.
    """
    t = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    rng = np.max(t[ok]) - np.min(t[ok])
    if rng < 1e-9:
        return float("nan")
    return float(np.mean(hi[ok] - lo[ok]) / rng)


def reliability_curve(y_true, y_pred_quantiles, quantiles):
    """Empirical vs nominal coverage, for reliability diagrams.

    Parameters
    ----------
    y_true : array, shape (n,)
    y_pred_quantiles : array, shape (n, n_quantiles)
    quantiles : sequence of float, length n_quantiles

    Returns
    -------
    (nominal, empirical) : two 1-D arrays.
        A perfectly calibrated model lies on the diagonal. Points below
        the diagonal mean the intervals are too narrow — the model is
        overconfident.
    """
    t = np.asarray(y_true, dtype=float).ravel()
    q_pred = np.asarray(y_pred_quantiles, dtype=float)
    q_levels = np.asarray(quantiles, dtype=float).ravel()

    if q_pred.ndim != 2:
        raise ValueError(f"y_pred_quantiles must be 2-D, got {q_pred.ndim}-D")
    if q_pred.shape[0] != t.shape[0]:
        raise ValueError(
            "y_true and y_pred_quantiles disagree on sample count"
        )
    if q_pred.shape[1] != q_levels.shape[0]:
        raise ValueError("y_pred_quantiles column count != len(quantiles)")

    empirical = np.full(q_levels.shape, np.nan, dtype=float)
    for i in range(q_levels.size):
        ok = np.isfinite(t) & np.isfinite(q_pred[:, i])
        if ok.any():
            empirical[i] = float(np.mean(t[ok] <= q_pred[ok, i]))
    return q_levels, empirical


# --------------------------------------------------------------------------
# Convenience bundle
# --------------------------------------------------------------------------


def point_metrics(y_true, y_pred, y_reference=None) -> dict[str, float]:
    """Every deterministic metric in one dict, for building results tables.

    ``y_reference`` should be the smart-persistence forecast; when
    omitted, the forecast-skill entry is NaN rather than absent, so that
    results frames keep a stable column set across strata.
    """
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mbe": mbe(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "nmae": nmae(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "n": int(np.sum(np.isfinite(np.asarray(y_true, dtype=float).ravel()))),
    }
    out["forecast_skill"] = (
        forecast_skill(y_true, y_pred, y_reference)
        if y_reference is not None
        else float("nan")
    )
    return out
