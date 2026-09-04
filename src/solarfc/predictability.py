"""Bounds on forecast error, after Yang's predictability framework.

Every model in this project has now hit the same wall. Extra features,
seven times the training data, deeper trees, a different algorithm and
targeted variability features all moved nothing. That is a claim about
the *atmosphere* rather than about any model, and it should be measured
directly instead of inferred from a list of things that failed.

This module implements the measurement. Forecast error is bracketed:

* an **upper bound**, the RMSE of the best reference forecast anyone
  should accept -- the optimal convex combination of climatology and
  persistence, known as clear-sky CLIPER. A forecast worse than this is
  worse than doing nothing clever.
* a **lower bound**, the error growth of a dynamical system started from
  slightly different initial conditions. No forecast can beat it,
  because it is the atmosphere's own divergence rate.

The gap between them is what forecasting can actually win, and
``predictability`` expresses it as a number in [0, 1].

Method
------
Following Liu and Yang (2023), *Renewable and Sustainable Energy
Reviews* 182:113359, who mapped this over the contiguous United States.
Their equation numbers are cited against each function.

The upper bound rests on a small result: for a stationary series, the
MSE of the optimal climatology-persistence blend depends on the lag-h
autocorrelation alone,

    MSE_kappa = (1 - rho_h^2) * V(kappa),

and scales to irradiance by the mean square clear-sky irradiance,

    A_r^2 = E(c^2) * MSE_kappa.                              (Eq. 4)

Since data exist only at discrete lags, a parametric correlogram is
fitted to the empirical autocorrelations so the bound can be evaluated
at any horizon (Eq. 5).

The lower bound uses the spread between a control forecast and its
perturbed siblings -- predictability error growth. Mean square PEG is
linear in horizon (Eq. 8), and its intercept is fixed by the nugget from
the correlogram fit rather than estimated freely, so **the upper bound
must be fitted first**.

Two things this project can do that the CONUS study could not
-------------------------------------------------------------
The data here is **10-minute**, where Yang used hourly. The nugget is a
property of the correlogram as the lag approaches zero, so an hourly
series can only extrapolate to it while a 10-minute series resolves it:
fitted from lags of 1 h and up the nugget collapses onto zero, and from
10-minute lags it lands at 0.046-0.078, worth 59-80 W/m^2.

The sites are **equatorial and monsoon-stratified**, which is the gap in
the literature. An OpenAlex sweep returns eight works at the
intersection of predictability and solar forecast skill, and four for
monsoon and solar predictability, none of them relevant.

On the diurnal bumps
--------------------
The empirical autocorrelation is not monotone. It decays to roughly zero
by 9-10 h, dips slightly negative where daytime samples pair with night,
then rebounds at 24 h and 48 h because the same clock time a day later
is meteorologically similar. Pair counts swing with it, from 124,000 at
a 24 h lag down to **none at all** at 12 h and 36 h, where no daytime
sample has a daytime partner.

This is expected and it is why the correlogram is fitted rather than
used empirically: a smooth parametric curve passes through the bumps and
can be evaluated at horizons that have no valid pairs at all. Yang's
Figure 2 shows the same rebounds over CONUS and treats them the same
way. Night samples must be held as gaps rather than dropped, so that
the spacing between observations stays true -- dropping them silently
renumbers every lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CauchyCorrelogram",
    "ErrorGrowth",
    "autocorrelation",
    "fit_correlogram",
    "upper_bound_rmse",
    "mspeg",
    "fit_error_growth",
    "predictability",
]

#: Smallest number of overlapping pairs an autocorrelation estimate needs.
#: Lags that straddle the night have far fewer pairs than lags landing a
#: whole day apart, and at a 12 h lag on this grid there are none.
MIN_PAIRS = 2000


def autocorrelation(values, lag_steps: int) -> tuple[float, int]:
    """Lag-h autocorrelation over pairs where both ends are observed.

    Parameters
    ----------
    values : array-like
        Clear-sky index on its native grid, with night held as NaN
        rather than removed. The gaps carry the timing: dropping them
        would make lag 1 mean "the next daytime sample", which spans a
        night at every dusk.
    lag_steps : int
        Lag in grid steps.

    Returns
    -------
    tuple
        ``(rho, n_pairs)``. ``rho`` is NaN when fewer than
        :data:`MIN_PAIRS` pairs survive.
    """
    if lag_steps < 1:
        raise ValueError(f"lag_steps must be >= 1, got {lag_steps}")

    x = np.asarray(values, dtype=float).ravel()
    if lag_steps >= x.size:
        return float("nan"), 0

    a, b = x[:-lag_steps], x[lag_steps:]
    both = np.isfinite(a) & np.isfinite(b)
    n = int(both.sum())
    if n < MIN_PAIRS:
        return float("nan"), n
    return float(np.corrcoef(a[both], b[both])[0, 1]), n


@dataclass(frozen=True)
class CauchyCorrelogram:
    """Generalised Cauchy correlation function with a nugget effect.

    ``C(tau) = (1 - nugget) * [1 + (tau / scale)^alpha]^(-beta / alpha)``
    for ``tau > 0``, and ``C(0) = 1``.

    The nugget is the discontinuity at the origin: as the lag approaches
    zero the curve tends to ``1 - nugget``, but a series is perfectly
    correlated with itself, so the value at exactly zero jumps to 1. That
    step is the part of the variation too fast for the observation
    interval to see, and it sets the floor on any forecast's error.

    Attributes
    ----------
    nugget : float
        Height of the jump at the origin, in [0, 1).
    scale_hours : float
        Range parameter, in hours.
    alpha : float
        Smoothness near the origin, in (0, 2].
    beta : float
        Tail decay.
    """

    nugget: float
    scale_hours: float
    alpha: float
    beta: float

    def __call__(self, tau_hours):
        tau = np.asarray(tau_hours, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (1.0 - self.nugget) * np.power(
                1.0 + np.power(tau / self.scale_hours, self.alpha),
                -self.beta / self.alpha,
            )
        # A series is perfectly correlated with itself; the nugget is a
        # limit approached from the right, not a value attained.
        return np.where(tau == 0, 1.0, out)

    def nugget_rmse(self, variance_kappa: float, mean_sq_clearsky: float):
        """The nugget expressed as an irradiance RMSE, in W/m^2.

        This is the quantity Yang's Eq. (8) fixes the error-growth
        intercept to, so it is worth being able to read directly.
        """
        share = 1.0 - (1.0 - self.nugget) ** 2
        return float(np.sqrt(share * variance_kappa * mean_sq_clearsky))


def fit_correlogram(
    lag_hours, rho, *, initial=(0.1, 1.0, 1.0, 1.0)
) -> CauchyCorrelogram:
    """Least-squares fit of :class:`CauchyCorrelogram` to empirical rho.

    Include sub-hourly lags wherever the grid allows. The nugget is
    identified by how the curve behaves as the lag approaches zero, so
    fitting from 1 h upward leaves it unconstrained and the optimiser
    parks it on the boundary at zero.

    Non-finite pairs are dropped, which is what removes the lags that
    have no valid observations -- 12 h and 36 h on a daytime-masked
    series.
    """
    from scipy.optimize import curve_fit

    tau = np.asarray(lag_hours, dtype=float).ravel()
    r = np.asarray(rho, dtype=float).ravel()
    if tau.shape != r.shape:
        raise ValueError(f"shape mismatch: {tau.shape} vs {r.shape}")

    ok = np.isfinite(tau) & np.isfinite(r) & (tau > 0)
    if ok.sum() < 8:
        raise ValueError(f"only {int(ok.sum())} usable lags to fit")

    def model(t, nugget, scale, alpha, beta):
        return (1.0 - nugget) * np.power(
            1.0 + np.power(t / scale, alpha), -beta / alpha
        )

    (nugget, scale, alpha, beta), _ = curve_fit(
        model,
        tau[ok],
        r[ok],
        p0=list(initial),
        bounds=([0.0, 0.01, 0.05, 0.05], [0.95, 500.0, 2.0, 20.0]),
        maxfev=60000,
    )
    return CauchyCorrelogram(
        nugget=float(nugget),
        scale_hours=float(scale),
        alpha=float(alpha),
        beta=float(beta),
    )


def upper_bound_rmse(
    correlogram: CauchyCorrelogram,
    tau_hours,
    variance_kappa: float,
    mean_sq_clearsky: float,
):
    """RMSE of clear-sky CLIPER, the highest tolerable error (Eq. 5).

    ``A_r(tau) = {[1 - C(tau)^2] * V(kappa) * E(c^2)}^(1/2)``

    A forecast whose RMSE exceeds this is beaten by the optimal blend of
    climatology and persistence, so the operator is better off without
    it. Because the reference improves as the horizon shortens, the
    bound tightens sharply at short lead times and saturates once the
    autocorrelation has decayed.

    Parameters
    ----------
    variance_kappa : float
        Variance of the clear-sky index over daytime samples.
    mean_sq_clearsky : float
        ``E(c^2)``, the mean square clear-sky irradiance over the same
        samples. This is what carries the bound from clear-sky-index
        units into W/m^2.
    """
    c = np.asarray(correlogram(tau_hours), dtype=float)
    return np.sqrt(
        np.clip(1.0 - c**2, 0.0, None) * variance_kappa * mean_sq_clearsky
    )


def mspeg(control, perturbed) -> float:
    """Mean square predictability error growth (Eq. 6).

    ``(1/n) sum_t [ (1/m) sum_i (x_t^control - x_{t,i}^perturbed)^2 ]``

    The spread between a control forecast and its perturbed siblings.
    Both arrive from the same model at the same valid time, so the
    difference is not forecast error against an observation -- it is how
    fast two almost identical atmospheres diverge, which is the floor
    under any forecast of either.

    Parameters
    ----------
    control : array-like, shape (n,)
        Control forecast, in clear-sky-index units.
    perturbed : array-like, shape (n, m)
        Perturbed members aligned to the same valid times.
    """
    c = np.asarray(control, dtype=float).ravel()
    p = np.asarray(perturbed, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    if p.shape[0] != c.size:
        raise ValueError(
            f"shape mismatch: control {c.size}, members {p.shape}"
        )

    squared = (p - c[:, None]) ** 2
    per_time = np.nanmean(squared, axis=1)
    return float(np.nanmean(per_time))


@dataclass(frozen=True)
class ErrorGrowth:
    """Linear fit of mean square PEG against horizon (Eq. 8).

    ``MSPEG_kappa(tau) = slope * tau + [1 - (1 - nugget)^2] * V(kappa)``

    The intercept is *not* fitted. It is the nugget carried over from
    the correlogram, on the reasoning that error at zero lead is the
    initial-condition uncertainty, which is the same unresolved
    small-scale variation the nugget measures. So the upper bound has to
    be estimated first.
    """

    slope: float
    nugget: float
    variance_kappa: float

    def intercept(self) -> float:
        return (1.0 - (1.0 - self.nugget) ** 2) * self.variance_kappa

    def lower_bound_rmse(self, tau_hours, mean_sq_clearsky: float):
        """RMSE no forecast can beat (Eq. 9), in W/m^2."""
        tau = np.asarray(tau_hours, dtype=float)
        return np.sqrt(
            np.clip(self.slope * tau + self.intercept(), 0.0, None)
            * mean_sq_clearsky
        )


def fit_error_growth(
    tau_hours,
    mspeg_kappa,
    *,
    correlogram: CauchyCorrelogram,
    variance_kappa: float,
) -> ErrorGrowth:
    """Fit the slope of mean square PEG, intercept fixed by the nugget.

    Only the slope is free, so this is a one-parameter least-squares fit
    through a known intercept rather than an ordinary regression.
    """
    tau = np.asarray(tau_hours, dtype=float).ravel()
    y = np.asarray(mspeg_kappa, dtype=float).ravel()
    if tau.shape != y.shape:
        raise ValueError(f"shape mismatch: {tau.shape} vs {y.shape}")

    ok = np.isfinite(tau) & np.isfinite(y)
    if ok.sum() < 2:
        raise ValueError(f"only {int(ok.sum())} usable points to fit")

    intercept = (1.0 - (1.0 - correlogram.nugget) ** 2) * variance_kappa
    # Least squares through a fixed intercept: minimise ||a*tau - (y - b)||.
    t, resid = tau[ok], y[ok] - intercept
    slope = float(np.dot(t, resid) / np.dot(t, t))
    return ErrorGrowth(
        slope=slope,
        nugget=correlogram.nugget,
        variance_kappa=variance_kappa,
    )


def predictability(lower_rmse, upper_rmse):
    """``P = 1 - A_p / A_r`` (Eq. 1), bounded to [0, 1].

    Zero means the best possible forecast is no better than climatology
    blended with persistence, so there is nothing to win. One means the
    atmosphere is perfectly predictable at that horizon. Everything real
    sits in between, and the value is the share of the reference's error
    that a forecast could in principle remove.
    """
    lower = np.asarray(lower_rmse, dtype=float)
    upper = np.asarray(upper_rmse, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - lower / upper
    return np.clip(np.where(np.isfinite(out), out, np.nan), 0.0, 1.0)
