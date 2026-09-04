"""Fit the clear-sky-index correlogram and the upper bound on forecast error.

Run once; the output is committed, like the turbidity fit.

This is the half of Yang's predictability framework that needs no
download. The empirical lag-h autocorrelation of the clear-sky index
gives the MSE of clear-sky CLIPER -- the optimal blend of climatology
and persistence -- and that is the highest error any forecast should be
allowed to post. A model that does worse is beaten by the reference.

The lower bound comes from the TIGGE ensemble and is fitted separately,
using the nugget produced here as its intercept.

Why sub-hourly lags are included
--------------------------------
The nugget is the discontinuity as the lag goes to zero, and Yang's
CONUS study could only approach it from 1 h because the data was
hourly. On this 10-minute grid the first six lags resolve it directly.
Fitted from 1 h upward the nugget collapses onto the boundary at zero at
all seven sites; fitted from 10 minutes it lands at 0.046-0.078.

    python scripts/fit_predictability.py
    python scripts/fit_predictability.py --sites kuala_lumpur

Takes about a minute for all seven sites.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solarfc.baselines import clear_sky_index
from solarfc.clearsky import clearsky_ghi, load_turbidity
from solarfc.config import (
    DAYTIME_CLEARSKY_FLOOR,
    HORIZON_LABELS,
    HORIZON_STEPS,
    PROCESSED_DIR,
    SITE_KEYS,
    STEP_MINUTES,
)
from solarfc.data import load_site
from solarfc.predictability import (
    autocorrelation,
    fit_correlogram,
    upper_bound_rmse,
)

OUT_DIR = PROCESSED_DIR / "predictability"

#: Lags to fit, in grid steps. The first six are sub-hourly and are what
#: identify the nugget; the rest run hourly out past the longest horizon.
FIT_LAG_STEPS: tuple[int, ...] = tuple(range(1, 7)) + tuple(
    h * 6 for h in range(1, 103)
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sites", nargs="+", default=list(SITE_KEYS))
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args(argv)


def fit_site(site: str) -> dict:
    """Correlogram and bound for one site, plus the empirical curve."""
    frame = load_site(site)
    envelope = clearsky_ghi(frame.index, site).to_numpy()
    ghi = frame["GHI"].to_numpy(dtype=float)
    csi = clear_sky_index(ghi, envelope)

    # Night is held as NaN rather than dropped, so lags keep their real
    # spacing. See solarfc.predictability.
    daytime = (
        np.isfinite(envelope)
        & (envelope > DAYTIME_CLEARSKY_FLOOR)
        & np.isfinite(csi)
    )
    masked = np.where(daytime, csi, np.nan)

    variance_kappa = float(np.nanvar(masked))
    mean_sq_clearsky = float(np.mean(envelope[daytime] ** 2))

    lags_hours, rhos, pairs = [], [], []
    for steps in FIT_LAG_STEPS:
        rho, n = autocorrelation(masked, steps)
        lags_hours.append(steps * STEP_MINUTES / 60.0)
        rhos.append(rho)
        pairs.append(n)

    correlogram = fit_correlogram(lags_hours, rhos)
    horizons_h = [s * STEP_MINUTES / 60.0 for s in HORIZON_STEPS]
    bound = upper_bound_rmse(
        correlogram, horizons_h, variance_kappa, mean_sq_clearsky
    )

    return {
        "site": site,
        "daytime_samples": int(daytime.sum()),
        "variance_kappa": variance_kappa,
        "mean_sq_clearsky": mean_sq_clearsky,
        "correlogram": {
            "model": "generalised Cauchy with nugget",
            "nugget": correlogram.nugget,
            "scale_hours": correlogram.scale_hours,
            "alpha": correlogram.alpha,
            "beta": correlogram.beta,
        },
        "nugget_rmse_w_m2": correlogram.nugget_rmse(
            variance_kappa, mean_sq_clearsky
        ),
        "empirical": {
            "lag_hours": lags_hours,
            "rho": [None if not np.isfinite(r) else r for r in rhos],
            "n_pairs": pairs,
        },
        "upper_bound_rmse_w_m2": {
            label: float(value) for label, value in zip(HORIZON_LABELS, bound)
        },
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if not load_turbidity():
        print("ERROR: no fitted turbidity. Run scripts/fit_turbidity.py.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    for site in args.sites:
        results.append(fit_site(site))
        print(f"  fitted {site}", flush=True)

    path = args.out_dir / "correlograms.json"
    path.write_text(
        json.dumps(
            {
                "method": (
                    "Liu and Yang (2023), Renewable and Sustainable Energy "
                    "Reviews 182:113359, Eqs. 4-5. Upper bound only; the "
                    "lower bound needs the TIGGE ensemble."
                ),
                "lag_steps_fitted": list(FIT_LAG_STEPS),
                "step_minutes": STEP_MINUTES,
                "sites": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {path}")

    print("\nCorrelogram fits")
    print(
        f"{'site':<16}{'nugget':>8}{'scale h':>9}{'alpha':>7}"
        f"{'beta':>7}{'nugget RMSE':>13}"
    )
    for r in results:
        c = r["correlogram"]
        print(
            f"{r['site']:<16}{c['nugget']:>8.4f}{c['scale_hours']:>9.3f}"
            f"{c['alpha']:>7.2f}{c['beta']:>7.2f}"
            f"{r['nugget_rmse_w_m2']:>10.0f} W/m2"
        )

    print(
        "\nUpper bound on RMSE, W/m2 -- worse than this is worse than CLIPER"
    )
    print(f"{'site':<16}" + "".join(f"{h:>8}" for h in HORIZON_LABELS))
    for r in results:
        row = r["upper_bound_rmse_w_m2"]
        print(
            f"{r['site']:<16}"
            + "".join(f"{row[h]:>8.0f}" for h in HORIZON_LABELS)
        )

    minutes, secs = divmod(int(time.time() - started), 60)
    print(f"\ntotal {minutes}m{secs:02d}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
