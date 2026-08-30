"""Supervised-matrix assembly: one (X, y) pair per horizon, track and target.

This is where the encoder side (:mod:`solarfc.features`, everything at or before
the forecast origin ``t``) meets the decoder side (:mod:`solarfc.covariates`,
everything legitimately knowable about ``t + h`` at issue time). It is the only
module in the package that shifts a series forwards, so it is the only place a
leakage bug can originate.

Alignment convention
--------------------
Every row is indexed by its **forecast origin** ``t``. For a horizon of ``h``
steps:

* observed-past columns are read at ``t`` -- no shift;
* known-future columns are read at ``t + h`` -- ``shift(-h)``, and carry a
  ``kf_`` prefix;
* the target is read at ``t + h`` -- ``shift(-h)``.

So a row says: *standing at ``t``, with this history and this weather forecast,
the irradiance at ``t + h`` was ``y``.* The last ``h`` rows have no target and
are dropped.

Target representation
---------------------
Two targets are supported and both are trained in Phase 2, because the plan
specifies the clear-sky index as an input and as Module B's attention signal but
never states the output units (Module D lists horizons and quantiles only). The
choice is therefore measured rather than asserted, and frozen in
``config.py`` before Phase 3 so the Transformer work does not fork.

``ghi``
    Predict W/m^2 directly. Loss and reported metric share a space.
``csi``
    Predict the clear-sky index, then rescale by Ineichen clear-sky GHI at the
    target time. Stationarises the target and puts the model in the same
    representation as smart persistence, which makes Forecast Skill a
    like-for-like comparison.

Either way the *scored* quantity is GHI in W/m^2, via :func:`to_ghi`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DAYTIME_CLEARSKY_FLOOR, STEP_MINUTES
from .covariates import (
    NWP_FEATURES,
    TRACKS,
    build_known_future,
    known_future_columns,
    load_fitted_error_models,
)
from .features import assert_no_leakage, feature_columns
from .splits import split_label

__all__ = [
    "TARGETS",
    "KF_PREFIX",
    "SupervisedSet",
    "build_known_future_grid",
    "build_supervised",
    "expected_columns",
    "nwp_columns_in",
    "to_ghi",
    "track_seed",
]

#: Output representations compared in Phase 2.
TARGETS: tuple[str, ...] = ("ghi", "csi")

#: Prefix marking a column evaluated at the target timestamp rather than the
#: origin. A leakage audit is then a string check.
KF_PREFIX = "kf_"


@dataclass(frozen=True)
class SupervisedSet:
    """One assembled (X, y) pair, with everything needed to score it.

    ``clearsky_ghi`` is the Ineichen envelope at the *target* timestamp. It is
    carried separately from ``X`` because it is needed twice outside the model:
    to rescale a CSI prediction back to W/m^2, and to build the daytime mask
    that every metric is computed under.
    """

    X: pd.DataFrame
    y: pd.Series
    #: Ineichen clear-sky GHI at t + h, aligned to X.
    clearsky_ghi: pd.Series
    #: Observed GHI at t + h, aligned to X. Always in W/m^2 whatever the target.
    ghi: pd.Series
    #: 'train' / 'val' / 'test' / 'unused', by origin timestamp.
    split: pd.Series
    site: str
    horizon_steps: int
    track: str
    feature_set: str
    target: str

    def subset(self, split: str) -> "SupervisedSet":
        """Rows belonging to one chronological split."""
        mask = (self.split == split).to_numpy()
        return SupervisedSet(
            X=self.X.loc[mask],
            y=self.y.loc[mask],
            clearsky_ghi=self.clearsky_ghi.loc[mask],
            ghi=self.ghi.loc[mask],
            split=self.split.loc[mask],
            site=self.site,
            horizon_steps=self.horizon_steps,
            track=self.track,
            feature_set=self.feature_set,
            target=self.target,
        )

    def __len__(self) -> int:
        return len(self.X)


def track_seed(site: str, track: str, horizon_steps: int) -> int:
    """Deterministic RNG seed for the realistic track's degradation draw.

    The realistic track perturbs ERA5 with correlated noise. That noise must be
    identical every time a given (site, track, horizon) is rebuilt, or two runs
    of the same experiment would differ for reasons unrelated to the model --
    and the difference would be small enough to mistake for a real effect.

    Derived from a hash rather than a counter so that adding a site or a horizon
    does not renumber the others.
    """
    key = f"{site}|{track}|{horizon_steps}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def build_known_future_grid(
    index: pd.DatetimeIndex,
    site: str,
    track: str,
    era5: pd.DataFrame | None = None,
    *,
    lead_hours: float | None = None,
    error_models=None,
    rng=None,
) -> pd.DataFrame:
    """Known-future covariates on the native grid, prefixed with ``kf_``.

    Evaluated *at the timestamps given*, not at an origin. The caller shifts.

    ``lead_hours`` is a scalar here rather than a per-row array: in a direct
    multi-horizon setup every row in a given matrix shares one lead time, which
    is exactly what makes the degradation well defined per model.
    """
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")

    leads = None
    if track == "realistic":
        if lead_hours is None:
            raise ValueError("track 'realistic' requires lead_hours")
        leads = np.full(len(index), float(lead_hours))

    frame = build_known_future(
        index,
        site,
        track=track,
        era5=era5,
        error_models=error_models,
        lead_hours=leads,
        rng=rng,
    )
    return frame.add_prefix(KF_PREFIX)


def build_supervised(
    features: pd.DataFrame,
    site: str,
    horizon_steps: int,
    *,
    track: str = "realistic",
    feature_set: str = "full",
    target: str = "csi",
    era5: pd.DataFrame | None = None,
    error_models=None,
    drop_night: bool = True,
    daytime_floor: float = DAYTIME_CLEARSKY_FLOOR,
    known_future: pd.DataFrame | None = None,
) -> SupervisedSet:
    """Assemble the supervised matrix for one horizon.

    Parameters
    ----------
    features : DataFrame
        Observed-past frame from :func:`solarfc.features.build_observed_past`.
    site : str
        Site key.
    horizon_steps : int
        Forecast horizon in steps of ``config.STEP_MINUTES``.
    track : {"nwp_free", "realistic", "perfect"}
        Known-future regime. See :mod:`solarfc.covariates`.
    feature_set : {"full", "deployable"}
    target : {"ghi", "csi"}
    era5 : DataFrame, optional
        Required for the realistic and perfect tracks.
    drop_night : bool
        Drop rows whose *target* timestamp is night, by the same clear-sky floor
        the metrics use. On by default: the training distribution then matches
        the evaluation distribution exactly, roughly halves the row count, and
        stops the model spending capacity on a value that solar geometry already
        determines. A deployed model clamps the night to zero from geometry,
        which is exact rather than learned.
    known_future : DataFrame, optional
        Pre-built known-future grid on the native index, ``kf_``-prefixed and
        **not yet shifted**. Building it runs pvlib's SPA over the full record,
        which dominates assembly cost, and the grid depends only on site, track
        and lead time -- so it is identical across targets, and across horizons
        for every track except ``realistic``. Passing it in lets a run over the
        full grid build 91 grids instead of 462.

        Only reuse a grid whose lead time matches: the ``realistic`` track's
        degradation is lead-dependent, so sharing one across horizons would
        apply a single error magnitude to all of them. The columns are validated
        against the track, but the lead time cannot be checked from the frame
        alone, so this is the caller's responsibility.

    Returns
    -------
    SupervisedSet

    Notes
    -----
    Rows containing any NaN in ``X`` or ``y`` are dropped. The first 144 steps
    of a site's record are lost to the 24-hour rolling windows and the last
    ``horizon_steps`` to the forward shift; both are expected and reported in
    the returned frame's ``attrs``.
    """
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")

    index = pd.DatetimeIndex(features.index)
    lead_hours = horizon_steps * STEP_MINUTES / 60.0

    # --- encoder side: no shift, and no kf_ column may appear here ----------
    past_columns = feature_columns(features, feature_set)
    assert_no_leakage(past_columns)
    past = features[past_columns]

    # --- decoder side: evaluated at t + h, then aligned back to origin t ----
    #
    # NWP fields stay in the deployable set: a forecast is a subscription, not a
    # satellite retrieval, so a plant can have one. Only the satellite-only
    # observed-past retrievals are removed, and that happens upstream in
    # ``feature_columns``.
    if known_future is None:
        models = load_fitted_error_models() if error_models is None else error_models
        known_future = build_known_future_grid(
            index,
            site,
            track,
            era5=era5,
            lead_hours=lead_hours,
            error_models=models,
            rng=np.random.default_rng(track_seed(site, track, horizon_steps)),
        )
    elif list(known_future.columns) != [
        f"{KF_PREFIX}{c}" for c in known_future_columns(track)
    ]:
        raise ValueError(
            f"supplied known_future does not match track {track!r}; expected "
            f"{[f'{KF_PREFIX}{c}' for c in known_future_columns(track)]}"
        )

    # A negative shift moves future values backwards onto the origin row, which
    # is the whole point -- and the one operation in this package that could
    # leak if its sign were wrong. The tests assert the alignment directly.
    known_future = known_future.reindex(index).shift(-horizon_steps)

    X = pd.concat([past, known_future], axis=1)

    # --- target and scoring series, both at t + h ---------------------------
    clearsky_future = features["clearsky_ghi_ineichen"].shift(-horizon_steps)
    ghi_future = features["ghi"].shift(-horizon_steps)

    if target == "ghi":
        y = ghi_future.rename("y")
    else:
        y = features["csi"].shift(-horizon_steps).rename("y")

    split = pd.Series(split_label(index), index=index, name="split")

    keep = np.isfinite(y.to_numpy()) & np.isfinite(clearsky_future.to_numpy())
    keep &= np.isfinite(ghi_future.to_numpy())
    keep &= X.notna().all(axis=1).to_numpy()

    if drop_night:
        keep &= clearsky_future.to_numpy() > daytime_floor

    X = X.loc[keep]
    result = SupervisedSet(
        X=X,
        y=y.loc[keep],
        clearsky_ghi=clearsky_future.loc[keep],
        ghi=ghi_future.loc[keep],
        split=split.loc[keep],
        site=site,
        horizon_steps=horizon_steps,
        track=track,
        feature_set=feature_set,
        target=target,
    )
    result.X.attrs.update(
        {
            "site": site,
            "horizon_steps": horizon_steps,
            "track": track,
            "feature_set": feature_set,
            "target": target,
            "rows_in": int(len(index)),
            "rows_kept": int(len(X)),
            "lead_hours": lead_hours,
        }
    )
    return result


def to_ghi(prediction, clearsky_ghi, target: str) -> np.ndarray:
    """Convert a model output to W/m^2, whatever representation it was trained in.

    Every metric in this project is computed on GHI in W/m^2, so this is the
    single point at which the two target representations become comparable.
    A CSI prediction is rescaled by the Ineichen envelope at the target time --
    the same envelope the target was divided by, so the operation is exactly
    invertible up to the clip applied when the index was formed.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    values = np.asarray(prediction, dtype=float).ravel()
    if target == "ghi":
        return values

    envelope = np.asarray(clearsky_ghi, dtype=float).ravel()
    if values.shape != envelope.shape:
        raise ValueError(
            f"shape mismatch: prediction {values.shape} vs clearsky {envelope.shape}"
        )
    # Irradiance cannot be negative, and a CSI prediction can be if the model
    # extrapolates. Clip after rescaling so the bound holds in the reported
    # units rather than in the index.
    return np.maximum(values * envelope, 0.0)


def nwp_columns_in(X: pd.DataFrame) -> list[str]:
    """Known-future NWP columns present in an assembled matrix.

    Used by the leakage tests: the ``nwp_free`` track must return an empty list.
    """
    return [c for c in X.columns if c in {f"{KF_PREFIX}{v}" for v in NWP_FEATURES}]


def expected_columns(features: pd.DataFrame, feature_set: str, track: str) -> list[str]:
    """Column list a matrix for this configuration should have, in order."""
    past = feature_columns(features, feature_set)
    return past + [f"{KF_PREFIX}{c}" for c in known_future_columns(track)]
