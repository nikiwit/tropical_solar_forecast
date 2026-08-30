"""Score the reference forecasts and write them into the results schema.

Every model in this project is measured against these numbers, so they have to
come from a committed script rather than from a notebook that has since moved
on. The reference table quoted in DOCS.md was produced interactively and could
not be regenerated; this replaces it.

Baselines scored:

    naive_persistence           forecast(t+h) = obs(t)
    smart_persistence_nsrdb     csi(t) * clearsky(t+h), NSRDB envelope
    smart_persistence_ineichen  csi(t) * clearsky(t+h), fitted Ineichen
    clearsky                    forecast(t+h) = clearsky(t+h)

Two smart-persistence variants, because Forecast Skill is a ratio against a
reference and the honest choice is the strongest one available. NSRDB's
envelope gives a reference 20-40 W/m^2 stronger at the horizons whose origins
fall at night (6h, 12h, 36h), so it is the primary FS reference and the
conservative bar. The Ineichen variant is what a plant could actually run,
having no satellite; the gap between the two is the clear-sky calibration
penalty, which is a reportable number for Contribution 4 rather than an
inconvenience.

Run after scripts/fit_turbidity.py -- the Ineichen variant and the clear-sky
baseline both depend on the calibrated envelope.

    python scripts/run_baselines.py
    python scripts/run_baselines.py --sites kuala_lumpur --splits test

Takes roughly 2-3 minutes for all seven sites.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from solarfc.baselines import clear_sky_index, naive_persistence, smart_persistence
from solarfc.clearsky import clearsky_ghi, load_turbidity
from solarfc.config import (
    HORIZON_LABELS,
    HORIZON_STEPS,
    RESULTS_DIR,
    SITE_KEYS,
    STEP_MINUTES,
)
from solarfc.data import load_site
from solarfc.results import RunMeta, append_results, score_predictions
from solarfc.splits import split_label

BASELINES = (
    "naive_persistence",
    "smart_persistence_nsrdb",
    "smart_persistence_ineichen",
    "clearsky",
)

#: The Forecast Skill reference quoted in headline tables. The strongest
#: available reference is the conservative choice: a weaker one inflates every
#: model's skill score.
PRIMARY_FS_REFERENCE = "smart_persistence_nsrdb"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(SITE_KEYS))
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--out", default=str(RESULTS_DIR / "results.csv"))
    parser.add_argument(
        "--overwrite", action="store_true", help="delete existing baseline rows first"
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    if not load_turbidity():
        print("ERROR: no fitted turbidity. Run scripts/fit_turbidity.py first.")
        return 1

    from pathlib import Path

    out_path = Path(args.out)
    if args.overwrite and out_path.exists():
        existing = pd.read_csv(out_path)
        kept = existing[~existing["model"].isin(BASELINES)]
        kept.to_csv(out_path, index=False)
        print(f"Removed {len(existing) - len(kept)} existing baseline rows\n")

    meta = {
        name: RunMeta(run_id=f"baselines-{int(time.time())}", model=name)
        for name in BASELINES
    }

    total = len(args.sites) * len(HORIZON_STEPS)
    done = 0
    started = time.time()
    print(f"Scoring {len(BASELINES)} baselines over {len(args.sites)} site(s) "
          f"x {len(HORIZON_STEPS)} horizons, splits {args.splits}\n")

    for site in args.sites:
        site_start = time.time()
        frame = load_site(site)
        index = frame.index
        ghi = frame["GHI"].to_numpy(dtype=float)
        envelope = clearsky_ghi(index, site).to_numpy()
        nsrdb_envelope = frame["Clearsky GHI"].to_numpy(dtype=float)
        splits = split_label(index)

        for steps, label in zip(HORIZON_STEPS, HORIZON_LABELS):
            predictions = {
                "naive_persistence": naive_persistence(ghi, steps),
                "smart_persistence_nsrdb": smart_persistence(
                    ghi, nsrdb_envelope, steps
                ),
                "smart_persistence_ineichen": smart_persistence(
                    ghi, envelope, steps
                ),
                "clearsky": envelope.copy(),
            }

            for split in args.splits:
                mask = splits == split
                if not mask.any():
                    continue
                for name, values in predictions.items():
                    finite = mask & np.isfinite(values)
                    if finite.sum() < 30:
                        continue
                    # The daytime mask comes from the fitted Ineichen envelope
                    # for every baseline, so all of them are scored on exactly
                    # the same samples. Letting each use its own envelope would
                    # give the NSRDB variant a different sample set and make the
                    # comparison between them meaningless.
                    scored = score_predictions(
                        ghi[finite],
                        values[finite],
                        envelope[finite],
                        index[finite],
                        reference_smart=predictions[PRIMARY_FS_REFERENCE][finite],
                        reference_naive=predictions["naive_persistence"][finite],
                    )
                    append_results(
                        scored,
                        out_path,
                        meta=meta[name],
                        site=site,
                        horizon_label=label,
                        horizon_steps=steps,
                        track="none",
                        feature_set="none",
                        target="ghi",
                        split=split,
                    )

            done += 1
            elapsed = time.time() - started
            eta = elapsed / done * (total - done)
            print(
                f"\r  [{done}/{total}] {site} {label:>6}   "
                f"elapsed {_duration(elapsed)}  ETA {_duration(eta)}   ",
                end="",
                flush=True,
            )

        print(f"\r  {site:>14} done in {_duration(time.time() - site_start)}"
              f"{' ' * 30}")

    print(f"\nWrote {out_path}  ({_duration(time.time() - started)} total)")
    _print_reference_table(out_path, args.sites[0])
    return 0


def _print_reference_table(path, site: str) -> None:
    """Print the smart-persistence numbers every later model is compared to."""
    from solarfc.results import load_results

    frame = load_results(path)
    ordered = {label: i for i, label in enumerate(HORIZON_LABELS)}

    def rows_for(model: str):
        subset = frame[
            (frame["model"] == model)
            & (frame["site"] == site)
            & (frame["split"] == "test")
            & (frame["stratum"] == "all")
        ]
        return subset.sort_values("horizon_label", key=lambda s: s.map(ordered))

    primary = rows_for(PRIMARY_FS_REFERENCE)
    deployable = rows_for("smart_persistence_ineichen")
    if primary.empty:
        return

    print(f"\nSmart persistence, {site}, test split (2020), daytime only.")
    print("NSRDB envelope is the primary FS reference; Ineichen is what a")
    print("plant could run without a satellite.\n")
    print(
        f"{'horizon':>8} {'MAE':>8} {'RMSE':>8} {'FS/naive':>9} | "
        f"{'MAE_ine':>8} {'RMSE_ine':>9} {'gap':>7}"
    )
    lookup = deployable.set_index("horizon_label")
    for _, row in primary.iterrows():
        label = row["horizon_label"]
        line = (
            f"{label:>8} {row['mae']:>8.1f} {row['rmse']:>8.1f} "
            f"{row['fs_naive']:>9.3f} | "
        )
        if label in lookup.index:
            other = lookup.loc[label]
            line += (
                f"{other['mae']:>8.1f} {other['rmse']:>9.1f} "
                f"{other['mae'] - row['mae']:>+7.1f}"
            )
        print(line)
    print("\nThe MAE column is the bar XGBoost has to clear.")


if __name__ == "__main__":
    raise SystemExit(main())
