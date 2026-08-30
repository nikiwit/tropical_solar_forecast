"""Site-calibrated clear-sky irradiance, and the Linke turbidity fit behind it.

Why the default climatology is not good enough
----------------------------------------------
pvlib's Ineichen model reads Linke turbidity from a global monthly climatology.
Measured against NSRDB's own clear-sky series over 2016-2020, that default runs
**4.4-7.6% low** at all seven study sites, with RMSE 27-35 W/m^2. The effect is
not subtle: observed GHI appears to exceed the clear-sky envelope for 19-28% of
daytime samples, which reads as an error to anyone who checks.

This is a known failure mode rather than a surprise. Chen et al. (2022,
*Renewable Energy*) report that using default model inputs instead of
locally measured parameters leads models to underestimate solar
radiation, and that estimating turbidity from local meteorological data
cuts clear-sky GHI RMSE from 24.02 to 9.94 W/m^2. Fitting turbidity to
the site is standard practice.

Fitting one value per site on the training years reduces RMSE against
NSRDB's clear-sky from 27.2 to 17.9 W/m^2 at Kuala Lumpur and 35.5 to
18.2 at Ho Chi Minh City. A per-month fit was tested and reaches only
17.1 -- an extra eleven parameters per site for the last 5% of the
improvement, with the added risk of absorbing genuine seasonal aerosol
variation into the clear-sky baseline, which is signal Module A is meant
to learn. Annual per-site is therefore the choice.

Why Ineichen rather than McClear or REST2
-----------------------------------------
Yang (2020), *Choice of clear-sky model in solar forecasting*, compares
Ineichen-Perez, McClear and REST2 and finds forecast RMSE is essentially the
same whichever is used -- complexity buys nothing downstream -- so the model
should be chosen on accessibility. Yang recommends McClear on those grounds, but
McClear is a web service. SolarInfer has to compute the clear-sky envelope on a
Raspberry Pi with no network, so Ineichen with a fitted turbidity constant is
the accessible option *here*: one extra float per site in the C++ engine.

What the clear-sky index means in this dataset
----------------------------------------------
NSRDB GHI never exceeds NSRDB clear-sky. The maximum ratio is exactly 1.000000
at every site, and 20-34% of daytime samples have GHI *identically equal* to the
clear-sky value. Solcast behaves the same way (maximum ratio exactly 1.000).
Both are satellite retrievals of the form ``GHI = clearsky x transmittance``
with transmittance bounded above by one, so **neither product can represent
cloud enhancement at all**.

Two consequences, both of which belong in the methodology chapter:

1. The clear-sky index here is retrieved cloud transmittance, not a measured
   over-irradiance ratio. Values above 1 against a *fitted* envelope are
   calibration residue, not physics.
2. The benchmark cannot evaluate cloud-enhancement events, which produce some of
   the sharpest positive ramps in tropical conditions. Ramp results
   therefore cover downward and moderate upward ramps only. This is a
   property of the evaluation target, not of any model tested.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, TRAIN_YEARS, Site, SITES_BY_KEY

__all__ = [
    "TURBIDITY_PATH",
    "DEFAULT_TURBIDITY_BOUNDS",
    "clearsky_ghi",
    "clear_turbidity_cache",
    "fit_linke_turbidity",
    "load_turbidity",
    "save_turbidity",
]

#: Fitted turbidity artefact. Committed -- it is a result, not raw data,
#: and the pipeline must be reproducible without re-running the fit.
TURBIDITY_PATH = PROCESSED_DIR / "clearsky" / "linke_turbidity.json"

#: Search bounds for the fit. Linke turbidity is physically ~1.5 (very
#: clean, high altitude) to ~8 (heavy urban haze); tropical maritime
#: sites sit around 3-5, so these bounds are permissive without being
#: meaningless.
DEFAULT_TURBIDITY_BOUNDS: tuple[float, float] = (1.5, 8.0)

#: Iterations of the detect-then-fit loop. Clear-sky detection needs a
#: clear-sky reference, which is what is being fitted, so the two are
#: alternated. It converges in two or three passes; five is a ceiling,
#: not an expectation.
MAX_FIT_ITERATIONS = 5

#: Convergence tolerance on the turbidity value between iterations.
FIT_TOLERANCE = 0.01

#: Window for Reno-Hansen clear-sky detection, in minutes.
#:
#: pvlib defaults to 10 minutes because the algorithm was designed for
#: 1-minute data, where that is ten samples. On this project's 10-minute
#: grid it would be a single sample and the algorithm refuses to run. 60
#: minutes gives six samples per window, which is the smallest span that
#: still lets the line-length and variability criteria discriminate a
#: clear hour from a smoothly-varying overcast one.
DETECT_WINDOW_MINUTES = 60


def _location(site: Site | str):
    import pvlib

    if isinstance(site, str):
        site = SITES_BY_KEY[site]
    return (
        pvlib.location.Location(
            latitude=site.latitude,
            longitude=site.longitude,
            altitude=site.elevation,
            tz="UTC",
        ),
        site,
    )


def clearsky_ghi(
    index: pd.DatetimeIndex,
    site: Site | str,
    turbidity: float | None = None,
) -> pd.Series:
    """Ineichen clear-sky GHI with the site's fitted Linke turbidity.

    Falls back to pvlib's climatology when no fit is available, so a
    fresh checkout still runs -- but it warns when it does. The fallback
    is 4.4-7.6% low, which is large enough to move every clear-sky index
    in the project and small enough to pass unnoticed in a results
    table, so it must not be silent. Run ``scripts/fit_turbidity.py``
    before producing anything reportable.
    """
    location, site_obj = _location(site)
    if turbidity is None:
        turbidity = load_turbidity().get(site_obj.key)

    if turbidity is None:
        warnings.warn(
            f"no fitted Linke turbidity for {site_obj.key!r}; falling back to "
            f"pvlib's climatology, which runs 4.4-7.6% low at these sites. "
            f"Run scripts/fit_turbidity.py before reporting any result.",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = location.get_clearsky(index, model="ineichen")
    else:
        frame = location.get_clearsky(
            index, model="ineichen", linke_turbidity=float(turbidity)
        )
    return frame["ghi"]


def fit_linke_turbidity(
    ghi: pd.Series,
    site: Site | str,
    *,
    years=TRAIN_YEARS,
    bounds: tuple[float, float] = DEFAULT_TURBIDITY_BOUNDS,
    progress=None,
) -> dict:
    """Fit one Linke turbidity value to a site's observed clear-sky periods.

    The turbidity that best reproduces measured irradiance on genuinely
    clear timesteps is what a plant would fit from its own pyranometer
    history, which is what keeps the calibration inside the DEPLOYABLE
    story.

    Clear timesteps are found with
    :func:`pvlib.clearsky.detect_clearsky`, the Reno-Hansen algorithm.
    That needs a clear-sky reference, which is the thing being fitted,
    so detection and fitting alternate until the turbidity stops moving.

    Parameters
    ----------
    ghi : Series
        Observed GHI, UTC-indexed on a uniform grid, spanning ``years``.
    site : Site or str
    years : tuple of int
        Restricted to the training years. Fitting on all five would let
        the clear-sky envelope -- and therefore every clear-sky index in
        the project -- carry information from the test year.

    Returns
    -------
    dict
        ``turbidity``, ``rmse``, ``n_clear``, ``iterations``,
        ``converged``, and the default it is replacing, for the record.
    """
    import pvlib
    from scipy.optimize import minimize_scalar

    location, site_obj = _location(site)

    mask = np.isin(
        np.asarray(ghi.index.year, dtype=int), np.asarray(years, dtype=int)
    )
    if not mask.any():
        raise ValueError(
            f"{site_obj.key}: no rows in fit years {tuple(years)}"
        )
    observed = ghi.loc[mask].astype(float)
    index = pd.DatetimeIndex(observed.index)

    default = float(
        np.mean(
            pvlib.clearsky.lookup_linke_turbidity(
                index, site_obj.latitude, site_obj.longitude
            )
        )
    )

    turbidity = default
    detected = None
    converged = False
    used = 0

    for iteration in range(1, MAX_FIT_ITERATIONS + 1):
        used = iteration
        reference = location.get_clearsky(
            index, model="ineichen", linke_turbidity=turbidity
        )["ghi"]
        detected = pvlib.clearsky.detect_clearsky(
            observed, reference, window_length=DETECT_WINDOW_MINUTES
        )

        if int(detected.sum()) < 100:
            # Too few clear samples to fit against. Keep the previous
            # estimate rather than fitting to noise.
            break

        clear_index = index[detected.to_numpy()]
        clear_observed = observed.to_numpy()[detected.to_numpy()]

        def rmse_at(value: float) -> float:
            modelled = location.get_clearsky(
                clear_index, model="ineichen", linke_turbidity=float(value)
            )["ghi"].to_numpy()
            return float(np.sqrt(np.mean((modelled - clear_observed) ** 2)))

        result = minimize_scalar(rmse_at, bounds=bounds, method="bounded")
        updated = float(result.x)

        if progress is not None:
            progress(
                site_obj.key,
                iteration,
                updated,
                int(detected.sum()),
                rmse_at(updated),
            )

        if abs(updated - turbidity) < FIT_TOLERANCE:
            turbidity = updated
            converged = True
            break
        turbidity = updated

    reference = location.get_clearsky(
        index, model="ineichen", linke_turbidity=turbidity
    )["ghi"]
    clear = (
        detected.to_numpy()
        if detected is not None
        else np.zeros(len(index), bool)
    )
    rmse = (
        float(
            np.sqrt(
                np.mean(
                    (reference.to_numpy()[clear] - observed.to_numpy()[clear])
                    ** 2
                )
            )
        )
        if clear.any()
        else float("nan")
    )

    return {
        "site": site_obj.key,
        "turbidity": round(turbidity, 4),
        "default_turbidity": round(default, 4),
        "rmse_clear": round(rmse, 3),
        "n_clear": int(clear.sum()),
        "iterations": used,
        "converged": bool(converged),
        "fit_years": [int(y) for y in years],
    }


@lru_cache(maxsize=8)
def _load_turbidity_cached(path_str: str) -> tuple[tuple[str, float], ...]:
    """Read and cache the artefact.

    ``solar_geometry`` calls this for every horizon, track and target in
    the grid -- several hundred times per run -- and the file never
    changes during one. Returned as a tuple of pairs because
    ``lru_cache`` requires a hashable return value that a caller cannot
    mutate into the cache.
    """
    path = Path(path_str)
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        (site, float(entry["turbidity"]))
        for site, entry in sorted(raw.items())
    )


def load_turbidity(path: str | Path | None = None) -> dict[str, float]:
    """Fitted turbidity per site, or an empty dict if the fit has not been run.

    Call :func:`clear_turbidity_cache` after re-running the fit within a
    live session, otherwise the previous values are reused.
    """
    path = TURBIDITY_PATH if path is None else Path(path)
    return dict(_load_turbidity_cached(str(path)))


def clear_turbidity_cache() -> None:
    """Forget the cached artefact, so the next read picks up a new fit."""
    _load_turbidity_cached.cache_clear()


def save_turbidity(
    results: dict[str, dict], path: str | Path | None = None
) -> Path:
    """Write the fit artefact, keys sorted so the file is byte-stable."""
    path = Path(TURBIDITY_PATH if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
