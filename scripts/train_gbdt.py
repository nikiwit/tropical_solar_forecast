"""Train gradient-boosted trees across the Phase 2 grid.

The grid is (site x horizon x track x feature_set x target). For all seven
sites that is 7 x 11 x 3 x 2 = 462 models per target, and Phase 2 runs both
target representations to settle which one the rest of the project uses, so
924 for XGBoost.

The run is resumable. Results append to the CSV as each model finishes, and a
combination already present is skipped, so an interrupted run continues rather
than restarting. Use --overwrite to force a clean re-run.

    python scripts/train_gbdt.py --smoke                 # 1 site, 3 horizons
    python scripts/train_gbdt.py                         # full grid, xgboost
    python scripts/train_gbdt.py --algorithm lightgbm --targets csi

Prerequisites: scripts/fit_turbidity.py, and data/processed/era5/*.parquet
from scripts/build_era5_cache.py.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from solarfc.baselines import naive_persistence, smart_persistence
from solarfc.clearsky import load_turbidity
from solarfc.config import (
    HORIZON_LABELS,
    HORIZON_STEPS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SITE_KEYS,
)
from solarfc.covariates import TRACKS
from solarfc.data import load_site
from solarfc.dataset import TARGETS, build_known_future_grid, build_supervised
from solarfc.features import FEATURE_SETS, build_observed_past
from solarfc.models.gbdt import GBDTConfig, best_iteration, fit_gbdt, predict_ghi
from solarfc.results import RunMeta, append_results, load_results, score_predictions

SMOKE_HORIZONS = ("20min", "6h", "36h")


def _duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _completed(path: Path, model_name: str) -> set[tuple]:
    """Combinations already in the results file, so a rerun can skip them."""
    if not path.exists():
        return set()
    frame = load_results(path)
    frame = frame[frame["model"] == model_name]
    if frame.empty:
        return set()
    keys = ["site", "horizon_label", "track", "feature_set", "target"]
    return set(map(tuple, frame[keys].drop_duplicates().to_numpy()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--sites", nargs="+", default=list(SITE_KEYS))
    parser.add_argument("--horizons", nargs="+", default=list(HORIZON_LABELS))
    parser.add_argument("--tracks", nargs="+", default=list(TRACKS))
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--out", default=str(RESULTS_DIR / "results.csv"))
    parser.add_argument("--smoke", action="store_true", help="1 site, 3 horizons")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    if args.smoke:
        args.sites = args.sites[:1]
        args.horizons = list(SMOKE_HORIZONS)

    if not load_turbidity():
        print("ERROR: no fitted turbidity. Run scripts/fit_turbidity.py first.")
        return 1

    horizon_of = dict(zip(HORIZON_LABELS, HORIZON_STEPS))
    unknown = [h for h in args.horizons if h not in horizon_of]
    if unknown:
        print(f"ERROR: unknown horizons {unknown}; expected {list(HORIZON_LABELS)}")
        return 1

    config = GBDTConfig(
        algorithm=args.algorithm,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )
    meta = RunMeta(
        run_id=f"{args.algorithm}-{int(time.time())}",
        model=args.algorithm,
        hyperparameters=config.to_dict(),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        existing = load_results(out_path)
        kept = existing[existing["model"] != args.algorithm]
        kept.to_csv(out_path, index=False)
        print(f"Removed {len(existing) - len(kept)} existing {args.algorithm} rows")

    done_already = _completed(out_path, args.algorithm)
    combos = [
        (site, label, track, feature_set, target)
        for site in args.sites
        for label in args.horizons
        for track in args.tracks
        for feature_set in args.feature_sets
        for target in args.targets
    ]
    todo = [c for c in combos if c not in done_already]

    print(f"{args.algorithm}: {len(combos)} combinations, {len(done_already)} already "
          f"done, {len(todo)} to train")
    print(f"  sites={len(args.sites)} horizons={len(args.horizons)} "
          f"tracks={args.tracks} feature_sets={args.feature_sets} targets={args.targets}")
    print(f"  writing to {out_path}\n")
    if not todo:
        print("Nothing to do.")
        return 0

    meta.to_json(PROCESSED_DIR / "runs" / f"{meta.run_id}.json")

    started = time.time()
    finished = 0
    failures: list[tuple] = []

    # Grouped by site so the NSRDB load and feature build happen once each, and
    # by track so the known-future grid is shared across horizons where it can
    # be. Rebuilding it per combination costs 8.5x on assembly alone.
    for site in args.sites:
        site_combos = [c for c in todo if c[0] == site]
        if not site_combos:
            continue

        load_start = time.time()
        nsrdb = load_site(site)
        era5_path = PROCESSED_DIR / "era5" / f"{site}_10min.parquet"
        if not era5_path.exists():
            print(f"ERROR: missing {era5_path}. Run scripts/build_era5_cache.py first.")
            return 1
        era5 = pd.read_parquet(era5_path)
        features = build_observed_past(nsrdb, site, era5=era5)

        ghi = nsrdb["GHI"].to_numpy(dtype=float)
        envelope = features["clearsky_ghi_ineichen"].to_numpy()
        nsrdb_envelope = nsrdb["Clearsky GHI"].to_numpy(dtype=float)
        print(f"{site}: features built in {_duration(time.time() - load_start)} "
              f"({features.shape[1]} columns)")

        grid_cache: dict[tuple, pd.DataFrame] = {}

        for site_key, label, track, feature_set, target in site_combos:
            steps = horizon_of[label]
            model_start = time.time()

            # nwp_free and perfect are lead-independent, so one grid serves
            # every horizon. realistic is not: its degradation is a function of
            # lead time, so it needs one per horizon.
            cache_key = (track, steps if track == "realistic" else -1)
            if cache_key not in grid_cache:
                grid_cache[cache_key] = build_known_future_grid(
                    features.index,
                    site,
                    track,
                    era5=era5,
                    lead_hours=steps * 10 / 60.0,
                )
            known_future = grid_cache[cache_key]

            try:
                built = build_supervised(
                    features,
                    site,
                    steps,
                    track=track,
                    feature_set=feature_set,
                    target=target,
                    era5=era5,
                    known_future=known_future,
                )
                train = built.subset("train")
                validation = built.subset("val")
                test = built.subset("test")
                if min(len(train), len(validation), len(test)) < 100:
                    raise ValueError(
                        f"insufficient rows: train={len(train)} val={len(validation)} "
                        f"test={len(test)}"
                    )

                model = fit_gbdt(train, validation, config)

                reference_naive = naive_persistence(ghi, steps)
                reference_smart = smart_persistence(ghi, nsrdb_envelope, steps)
                positions = features.index.get_indexer(test.X.index)

                predictions = predict_ghi(model, test)
                scored = score_predictions(
                    test.ghi.to_numpy(),
                    predictions,
                    test.clearsky_ghi.to_numpy(),
                    test.X.index,
                    reference_smart=reference_smart[positions + steps],
                    reference_naive=reference_naive[positions + steps],
                )
                append_results(
                    scored,
                    out_path,
                    meta=meta,
                    site=site,
                    horizon_label=label,
                    horizon_steps=steps,
                    track=track,
                    feature_set=feature_set,
                    target=target,
                    split="test",
                )

                aggregate = scored[scored["stratum"] == "all"]
                mae = float(aggregate["mae"].iloc[0]) if not aggregate.empty else np.nan
                skill = float(aggregate["fs_smart"].iloc[0]) if not aggregate.empty else np.nan
                status = (
                    f"MAE {mae:7.1f}  FS {skill:+.3f}  "
                    f"{best_iteration(model):>4} rounds"
                )
            except Exception as error:  # noqa: BLE001 - a failure must not stop the grid
                failures.append((site, label, track, feature_set, target, str(error)))
                status = f"FAILED: {error}"

            finished += 1
            elapsed = time.time() - started
            eta = elapsed / finished * (len(todo) - finished)
            print(
                f"  [{finished:>4}/{len(todo)}] {label:>6} {track:<9} "
                f"{feature_set:<10} {target:<3}  {status}   "
                f"({_duration(time.time() - model_start)}, ETA {_duration(eta)})",
                flush=True,
            )

    print(f"\nTrained {finished - len(failures)}/{len(todo)} in "
          f"{_duration(time.time() - started)}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for entry in failures[:20]:
            print(f"  {entry[:5]}  {entry[5]}")
        return 1

    _summarise(out_path, args.algorithm)
    return 0


def _summarise(path: Path, model_name: str) -> None:
    """Compare the two target representations, which is what Phase 2 decides."""
    frame = load_results(path)
    subset = frame[
        (frame["model"] == model_name)
        & (frame["split"] == "test")
        & (frame["stratum"] == "all")
    ]
    if subset.empty or subset["target"].nunique() < 2:
        return

    table = subset.pivot_table(index="target", values=["mae", "rmse", "fs_smart"])
    print("\nTarget representation, averaged over the grid:\n")
    print(table.round(3).to_string())
    best = table["mae"].idxmin()
    print(f"\nLower MAE: {best}. Freeze this in config before Phase 3.")


if __name__ == "__main__":
    raise SystemExit(main())
