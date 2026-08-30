"""Ramp-event detection and scoring.

Why this module exists
----------------------
RMSE is the wrong primary metric for this project's stated motivation. Grid
instability is not caused by average error; it is caused by *ramps* — the
60-90% irradiance collapse within minutes when an afternoon convective cell
arrives over an equatorial site.

Ramps are rare. RMSE is therefore dominated by the ~95% of timesteps
that are unremarkable, and a model can post an excellent RMSE while
missing every ramp that matters. Scoring ramps explicitly is what
connects the numbers to the smart-grid claim in the project title.

It is also the test Module A (monsoon-phase gating) must pass.
Convective ramps are precisely the tropical phenomenon the gate is
designed for, so if the gate works anywhere it should show up here — and
an RMSE-only evaluation would obscure it. If Tropical-TFT's margin over
the baselines is *not* larger on ramp events than on aggregate error,
that is evidence against Module A's rationale and must be reported as
such.

Definition
----------
A ramp occurs at time ``t`` when the change in GHI over a trailing window
exceeds a fraction of the concurrent clear-sky GHI:

    |GHI(t) - GHI(t - w)| > frac * clearsky_GHI(t)

Normalising by clear-sky GHI rather than using an absolute W/m^2
threshold makes the definition comparable across sites and across times
of day: a 200 W/m^2 drop at solar noon is routine, the same drop an hour
after sunrise is a total occlusion. The threshold and window are swept
(see ``RAMP_THRESHOLD_SWEEP``) so the headline choice is justified, not
asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import (
    DAYTIME_CLEARSKY_FLOOR,
    RAMP_THRESHOLD_FRAC,
    RAMP_WINDOW_STEPS,
    STEP_MINUTES,
)

__all__ = [
    "RampMetrics",
    "detect_ramps",
    "ramp_metrics",
    "ramp_detection_lead_time",
]


@dataclass(frozen=True)
class RampMetrics:
    """Scores for ramp detection at one horizon / stratum."""

    precision: float
    recall: float
    f1: float
    #: Mean lead time in minutes over correctly detected ramps. NaN when
    #: none.
    mean_lead_time_min: float
    n_observed: int
    n_predicted: int
    n_true_positive: int
    #: Fraction of daytime samples that are observed ramps. Reported
    #: because a very low base rate makes precision unstable and must
    #: temper reading of it.
    base_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def detect_ramps(
    ghi,
    clearsky_ghi,
    *,
    threshold_frac: float = RAMP_THRESHOLD_FRAC,
    window_steps: int = RAMP_WINDOW_STEPS,
    daytime_floor: float = DAYTIME_CLEARSKY_FLOOR,
    signed: bool = False,
):
    """Boolean array marking ramp events in a GHI series.

    Parameters
    ----------
    ghi, clearsky_ghi : array-like, shape (n,)
        Contiguous, evenly spaced series at ``STEP_MINUTES`` resolution.
    threshold_frac : float
        Ramp threshold as a fraction of concurrent clear-sky GHI.
    window_steps : int
        Trailing window over which the change is measured.
    daytime_floor : float
        Clear-sky GHI below which samples are night and cannot be ramps.
    signed : bool
        If True, return ``-1`` for down-ramps, ``+1`` for up-ramps and
        ``0`` otherwise, as an int array. Down-ramps are the
        operationally dangerous direction; up-ramps matter for
        curtailment.

    Returns
    -------
    ndarray
        Boolean mask, or int8 direction array when ``signed`` is True.
        The first ``window_steps`` entries are always non-events, since
        no trailing window exists for them.
    """
    if window_steps < 1:
        raise ValueError(f"window_steps must be >= 1, got {window_steps}")
    if not 0.0 < threshold_frac:
        raise ValueError(f"threshold_frac must be > 0, got {threshold_frac}")

    g = np.asarray(ghi, dtype=float).ravel()
    cs = np.asarray(clearsky_ghi, dtype=float).ravel()
    if g.shape != cs.shape:
        raise ValueError(
            f"shape mismatch: ghi {g.shape} vs clearsky {cs.shape}"
        )

    n = g.size
    delta = np.full(n, np.nan, dtype=float)
    if n > window_steps:
        delta[window_steps:] = g[window_steps:] - g[:-window_steps]

    # Night-time and non-finite samples can never be ramp events.
    is_day = np.isfinite(cs) & (cs > daytime_floor)
    threshold = threshold_frac * cs

    valid = is_day & np.isfinite(delta) & np.isfinite(threshold)
    is_ramp = np.zeros(n, dtype=bool)
    np.greater(np.abs(delta), threshold, out=is_ramp, where=valid)

    if not signed:
        return is_ramp

    direction = np.zeros(n, dtype=np.int8)
    direction[is_ramp & (delta < 0)] = -1
    direction[is_ramp & (delta > 0)] = 1
    return direction


def ramp_detection_lead_time(
    observed_ramp,
    predicted_ramp,
    *,
    tolerance_steps: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Match predicted ramps to observed ramps within a tolerance.

    Exact-timestep agreement is too strict to be meaningful: a forecast
    that calls a ramp one step early is operationally a success, not a
    miss. Each observed ramp is matched to the nearest unclaimed
    predicted ramp within +/- ``tolerance_steps``, greedily and
    one-to-one, so that a model spraying predictions cannot claim the
    same event twice.

    Returns
    -------
    (matched_observed_idx, lead_times_min)
        Indices of observed ramps that were detected, and the
        corresponding lead time in minutes. Positive lead time means the
        prediction came *before* the observed event.
    """
    obs_idx = np.flatnonzero(np.asarray(observed_ramp, dtype=bool).ravel())
    pred_idx = np.flatnonzero(np.asarray(predicted_ramp, dtype=bool).ravel())

    if obs_idx.size == 0 or pred_idx.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    claimed = np.zeros(pred_idx.size, dtype=bool)
    matched: list[int] = []
    leads: list[float] = []

    for o in obs_idx:
        offsets = pred_idx - o
        eligible = (np.abs(offsets) <= tolerance_steps) & ~claimed
        if not eligible.any():
            continue
        # Nearest eligible prediction; ties resolve to the earlier one,
        # which is the conservative choice for a lead-time claim.
        candidates = np.flatnonzero(eligible)
        best = candidates[np.argmin(np.abs(offsets[candidates]))]
        claimed[best] = True
        matched.append(int(o))
        # Prediction before the event => positive lead.
        leads.append(float(-offsets[best] * STEP_MINUTES))

    return np.asarray(matched, dtype=int), np.asarray(leads, dtype=float)


def ramp_metrics(
    y_true,
    y_pred,
    clearsky_ghi,
    *,
    threshold_frac: float = RAMP_THRESHOLD_FRAC,
    window_steps: int = RAMP_WINDOW_STEPS,
    tolerance_steps: int = 3,
    daytime_floor: float = DAYTIME_CLEARSKY_FLOOR,
) -> RampMetrics:
    """Precision / recall / F1 / mean lead time for ramp detection.

    Both the observed and the predicted ramp masks are derived with the
    *same* definition applied to the observed and forecast series
    respectively, so the comparison asks the operationally meaningful
    question: when the real world ramped, did the forecast series ramp
    too?

    Interpretation guidance for the results chapter:

    * **Recall** is the safety-critical number — a missed down-ramp is
      unserved load or an emergency reserve call.
    * **Precision** matters for trust — a model crying wolf gets switched off.
    * **Base rate** must be quoted alongside both: at a low base rate,
      precision is unstable and small absolute differences are noise.
    """
    obs = detect_ramps(
        y_true,
        clearsky_ghi,
        threshold_frac=threshold_frac,
        window_steps=window_steps,
        daytime_floor=daytime_floor,
    )
    pred = detect_ramps(
        y_pred,
        clearsky_ghi,
        threshold_frac=threshold_frac,
        window_steps=window_steps,
        daytime_floor=daytime_floor,
    )

    matched, leads = ramp_detection_lead_time(
        obs, pred, tolerance_steps=tolerance_steps
    )

    n_obs = int(obs.sum())
    n_pred = int(pred.sum())
    n_tp = int(matched.size)

    precision = n_tp / n_pred if n_pred else float("nan")
    recall = n_tp / n_obs if n_obs else float("nan")
    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and (precision + recall) > 0
    ):
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    cs = np.asarray(clearsky_ghi, dtype=float).ravel()
    n_day = int(np.sum(np.isfinite(cs) & (cs > daytime_floor)))

    return RampMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_lead_time_min=(
            float(np.mean(leads)) if leads.size else float("nan")
        ),
        n_observed=n_obs,
        n_predicted=n_pred,
        n_true_positive=n_tp,
        base_rate=float(n_obs / n_day) if n_day else float("nan"),
    )
