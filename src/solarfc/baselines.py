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

from .config import DAYTIME_CLEARSKY_FLOOR

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
    clip_max: float = 1.5,
):
    """CSI = GHI / clear-sky GHI, guarded at low sun and clipped above.

    Returns NaN where clear-sky GHI is at or below ``floor``: the ratio is
    meaningless at night and numerically explosive near sunrise and sunset.

    Values above 1 are physically real — cloud-edge enhancement genuinely
    exceeds the clear-sky envelope — so the clip is set at ``clip_max`` = 1.5
    rather than 1.0. Clipping at 1.0 would destroy exactly the over-irradiance
    events that matter for ramp analysis.
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
    """
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")

    cs = np.asarray(clearsky_ghi, dtype=float).ravel()
    csi = clear_sky_index(ghi, clearsky_ghi, floor=floor)
    origin_csi = _forward_fill(csi) if carry_overnight else csi

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
