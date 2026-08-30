"""Reference forecasts.

These are the models every other model is measured against, so they are built
before anything is trained.

The distinction that matters here is naive vs smart persistence. Naive
persistence — ``forecast(t+h) = obs(t)`` — is a weak reference in a strongly
diurnal signal: at a 12-hour horizon it predicts night-time irradiance from
midday values and is catastrophically wrong for reasons that have nothing to do
with weather. Forecast Skill computed against it therefore flatters every
model, sometimes dramatically.

Smart persistence holds the *clear-sky index* constant and rescales by clear-sky
irradiance at the target time, removing the deterministic solar geometry and
leaving only the atmospheric part to be predicted. It is the accepted reference
in the solar forecasting literature and is this project's primary FS reference.
Naive persistence is retained and reported only so that comparisons with papers
that use it remain possible.
"""

from __future__ import annotations

import numpy as np

from .config import CSI_CARRY_FLOOR, CSI_CLIP_MAX, DAYTIME_CLEARSKY_FLOOR

__all__ = [
    "clear_sky_index",
    "naive_persistence",
    "smart_persistence",
    "clear_sky_forecast",
]


def clear_sky_index(
    ghi,
    clearsky_ghi,
    *,
    floor: float = DAYTIME_CLEARSKY_FLOOR,
    clip_max: float = CSI_CLIP_MAX,
):
    """CSI = GHI / clear-sky GHI, guarded at low sun and clipped above.

    Returns NaN where clear-sky GHI is at or below ``floor``: the ratio is
    meaningless at night and numerically explosive near sunrise and sunset.

    On the clip
    -----------
    In a *measured* irradiance series, values above 1 are physically real:
    cloud-edge enhancement genuinely exceeds the clear-sky envelope, and
    clipping at 1.0 would destroy the over-irradiance events that matter for
    ramp analysis.

    That is not true of this project's data, and the difference is worth
    stating. NSRDB GHI never exceeds NSRDB clear-sky — the maximum ratio is
    exactly 1.000000 at every site, with 20–34% of daytime samples identically
    equal — and Solcast behaves the same way. Both are transmittance retrievals,
    ``GHI = clearsky * tau`` with ``tau <= 1``, so neither can represent
    enhancement at all. Against the *fitted* Ineichen envelope the index
    therefore stays within [0, 1] up to calibration residue.

    The clip is consequently set at ``config.CSI_CLIP_MAX`` = 2.0, comfortably
    above the fitted 99th percentile of 1.585, so it effectively never binds. It
    guards against a pathological twilight ratio reaching a squared loss; it is
    not truncating physics, because the physics it would truncate does not occur
    here. See :mod:`solarfc.clearsky`.
    """
    g = np.asarray(ghi, dtype=float).ravel()
    cs = np.asarray(clearsky_ghi, dtype=float).ravel()
    if g.shape != cs.shape:
        raise ValueError(f"shape mismatch: ghi {g.shape} vs clearsky {cs.shape}")

    out = np.full(g.shape, np.nan, dtype=float)
    ok = np.isfinite(g) & np.isfinite(cs) & (cs > floor)
    np.divide(g, cs, out=out, where=ok)
    return np.clip(out, 0.0, clip_max)


def naive_persistence(ghi, horizon_steps: int):
    """``forecast(t + h) = obs(t)``, aligned to the target timestamp.

    Element ``i`` of the result is the forecast *for* time ``i``, i.e. the
    observation from ``horizon_steps`` earlier. The first ``horizon_steps``
    entries are NaN because no origin exists for them.

    Reported for comparability with prior work only — see the module docstring.
    """
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")

    g = np.asarray(ghi, dtype=float).ravel()
    out = np.full(g.shape, np.nan, dtype=float)
    if g.size > horizon_steps:
        out[horizon_steps:] = g[:-horizon_steps]
    return out


def _forward_fill(values):
    """Carry the last finite value forward. Leading NaNs are left as NaN."""
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    if not ok.any():
        return v.copy()
    # Index of the most recent finite sample at or before each position.
    idx = np.maximum.accumulate(np.where(ok, np.arange(v.size), -1))
    out = np.where(idx >= 0, v[idx], np.nan)
    return out


def smart_persistence(
    ghi,
    clearsky_ghi,
    horizon_steps: int,
    *,
    floor: float = DAYTIME_CLEARSKY_FLOOR,
    carry_overnight: bool = True,
    carry_floor: float = CSI_CARRY_FLOOR,
):
    """``forecast(t + h) = csi(origin) * clearsky_ghi(t + h)``.

    The primary Forecast Skill reference for this project. Persisting the
    clear-sky index rather than raw irradiance removes the deterministic
    diurnal cycle, so the reference is only wrong to the extent that the
    *atmosphere* changed — which is what a forecast model is actually being
    asked to predict.

    The overnight problem
    ---------------------
    The instantaneous form ``csi(t)`` is undefined whenever the forecast origin
    falls at night, because CSI is a ratio against a clear-sky value of zero.
    At a 12-hour horizon on a 10-minute grid this is not an edge case: *every*
    daytime target has a night-time origin, so the entire horizon evaluates to
    NaN and drops out of the results table — silently removing the reference
    that Forecast Skill is measured against.

    With ``carry_overnight=True`` (the default) the origin CSI is the last
    *observed* clear-sky index at or before the origin, carried forward across
    the night. For a 12-hour-ahead forecast issued at midnight this means
    "assume tomorrow morning is as clear as yesterday afternoon was" — a real,
    defensible operational heuristic, and one that keeps the reference defined
    and comparable at every horizon.

    Set ``carry_overnight=False`` for the strict instantaneous definition, which
    is what some papers use; it is retained so published numbers that assume it
    can still be reproduced.

    Which value gets carried
    ------------------------
    Carrying the last CSI at *any* sun angle turns out to matter a great deal,
    because the clear-sky index is least trustworthy exactly where the carry
    starts. Twilight samples sit near the bottom of the clear-sky envelope, so
    small errors in that envelope produce large errors in the ratio -- the
    samples that hit the CSI clip have a median clear-sky of 38 W/m^2 against
    651 for daytime as a whole. Carrying one of those across the night and then
    multiplying it by a full midday clear-sky value scales the artefact up with
    the sun.

    Measured on KL 2020, carrying indiscriminately costs the reference 76 W/m^2
    of MAE at 6 h, 144 at 12 h and 147 at 36 h -- the three horizons whose
    origins are at night. A reference that weak would flatter every model
    measured against it, which is the failure the plan's warning about naive
    persistence exists to prevent.

    So only samples above ``carry_floor`` are eligible to be carried. Below it
    the instantaneous value is discarded in favour of the last reliable one.
    The threshold is not tuned: 100 W/m^2 is where the CSI clip was measured to
    stop binding altogether. The cost at short horizons, where the origin is
    already daylit, is 0.7 W/m^2 at 20 minutes.
    """
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")

    cs = np.asarray(clearsky_ghi, dtype=float).ravel()
    csi = clear_sky_index(ghi, clearsky_ghi, floor=floor)

    if carry_overnight:
        reliable = np.where(cs > carry_floor, csi, np.nan)
        origin_csi = _forward_fill(reliable)
    else:
        origin_csi = csi

    out = np.full(cs.shape, np.nan, dtype=float)
    if cs.size > horizon_steps:
        # CSI observed at the origin, clear-sky irradiance at the target time.
        out[horizon_steps:] = origin_csi[:-horizon_steps] * cs[horizon_steps:]

    # Night targets are genuinely zero, not unknown: clear-sky GHI at or below
    # the floor means the sun is down regardless of what the atmosphere did.
    night = np.isfinite(cs) & (cs <= floor)
    out[night] = 0.0
    return out


def clear_sky_forecast(clearsky_ghi):
    """Predict the clear-sky envelope itself — the physical upper bound.

    A pure-physics reference with no atmospheric information at all. In the
    tropics it over-predicts heavily, and that is the point: the gap between
    this and the observations is the share of the signal attributable to cloud,
    which is what every learned model in this project is competing to capture.
    """
    return np.asarray(clearsky_ghi, dtype=float).ravel().copy()
