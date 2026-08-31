"""Score the operational NWP baseline into the results schema.

This is the comparison that speaks to adoption. Every other reference in
the project is statistical or machine-learned; this one is the forecast a
plant is already receiving from a national met agency. AEMO accredits a
self-forecast only if it beats the incumbent on both MAE and RMSE, so
without this the project cannot make the claim that matters.

Three variants are scored, all driven by archived JMA GSM forecasts:

    nwp_jma_mos        ridge post-processing over every forecast field
    nwp_jma_cloud      fitted cloud-to-transmittance curve
    nwp_jma_cloud_kc   the same curve with Kasten-Czeplak's published
                       coefficients, for comparison only

MOS is the headline incumbent. Raw NWP output is post-processed before
anyone dispatches on it, and post-processing roughly doubles the
explained variance here, so scoring only the cloud-only form would
understate the incumbent and flatter everything measured against it.

Read the caveat in solarfc.nwp_baseline before quoting any of this: JMA
GSM carries no shortwave radiation, so the irradiance is derived from
forecast *cloud*. The cloud is a genuine forecast at a genuine lead time;
the conversion to irradiance is ours. Label it a cloud-driven NWP
baseline, never an operational GHI forecast.

Both the transmittance curve and the MOS coefficients are fitted on
training years only. The forecast archive starts in 2018, so in practice
they fit on 2018 and are evaluated on 2019 and 2020.

Run after scripts/fit_turbidity.py and scripts/pull_jma_forecasts.py:

    python scripts/run_nwp_baseline.py
    python scripts/run_nwp_baseline.py --sites kuala_lumpur --leads 24

Takes roughly a minute for all seven sites.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from solarfc.baselines import (
    clear_sky_index,
    naive_persistence,
    smart_persistence,
)
from solarfc.clearsky import clearsky_ghi, load_turbidity
from solarfc.config import (
    HORIZON_LABELS,
    HORIZON_STEPS,
    RESULTS_DIR,
    SITE_KEYS,
    TRAIN_YEARS,
)
from solarfc.data import load_site
from solarfc.nwp_baseline import (
    KASTEN_CZEPLAK,
    LEAD_HOURS,
    cloud_driven_ghi,
    fit_mos,
    fit_transmittance,
    lead_for_horizon,
    upsample_forecast,
)
from solarfc.nwp_error import load_jma
from solarfc.results import RunMeta, append_results, score_predictions
from solarfc.splits import split_label

MODELS = ("nwp_jma_mos", "nwp_jma_cloud", "nwp_jma_cloud_kc")

#: Matches scripts/run_baselines.py, so Forecast Skill here is on the
#: same reference as everywhere else in the results file.
PRIMARY_FS_REFERENCE = "smart_persistence_nsrdb"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(SITE_KEYS))
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=list(LEAD_HOURS),
        help="forecast lead offsets in hours; 0 is an analysis, not a "
        "forecast, and is scored only as a ceiling",
    )
    parser.add_argument("--out", default=str(RESULTS_DIR / "results.csv"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete existing NWP baseline rows first",
    )
    return parser.parse_args(argv)


def build_predictions(site: str, leads, index, envelope, observed_csi):
    """Fitted forecasts for one site, one series per model per lead.

    The prediction for a given lead does not depend on the horizon: a
    forecast valid at time T is a single value, however far ahead it was
    issued. Horizon enters only through which lead is admissible and
    through the persistence reference the forecast is scored against.
    That asymmetry is the whole reason NWP looks progressively better as
    the horizon grows -- it does not decay with lead time the way
    persistence does.
    """
    jma = load_jma(site)
    on_grid = upsample_forecast(jma, index)
    train = np.isin(np.asarray(index.year), TRAIN_YEARS)

    out: dict[tuple[str, int], np.ndarray] = {}
    fits: dict[int, dict] = {}

    for lead in leads:
        cloud_column = f"era5_cloud_cover_lead{lead}h"
        if cloud_column not in on_grid.columns:
            continue

        cloud = on_grid[cloud_column].to_numpy(dtype=float)
        usable = train & np.isfinite(cloud) & np.isfinite(observed_csi)
        if usable.sum() < 100:
            continue

        a, b = fit_transmittance(cloud[usable], observed_csi[usable])
        out[("nwp_jma_cloud", lead)] = cloud_driven_ghi(cloud, envelope, a, b)
        out[("nwp_jma_cloud_kc", lead)] = cloud_driven_ghi(
            cloud, envelope, *KASTEN_CZEPLAK
        )

        try:
            model = fit_mos(
                on_grid.loc[usable], observed_csi[usable], lead_hours=lead
            )
        except ValueError:
            model = None
        else:
            out[("nwp_jma_mos", lead)] = model.predict_ghi(on_grid, envelope)

        fits[lead] = {
            "a": a,
            "b": b,
            "n_fit": int(usable.sum()),
            "mos": model is not None,
        }

    return out, fits


def main(argv=None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore")

    if not load_turbidity():
        print("ERROR: no fitted turbidity. Run scripts/fit_turbidity.py.")
        return 1

    out_path = Path(args.out)
    if args.overwrite and out_path.exists():
        existing = pd.read_csv(out_path)
        kept = existing[~existing["model"].isin(MODELS)]
        kept.to_csv(out_path, index=False)
        print(f"Removed {len(existing) - len(kept)} existing NWP rows\n")

    run_id = f"nwp-baseline-{int(time.time())}"
    meta = {name: RunMeta(run_id=run_id, model=name) for name in MODELS}

    operational = {steps: lead_for_horizon(steps) for steps in HORIZON_STEPS}
    print(
        f"Scoring {len(MODELS)} NWP variants over {len(args.sites)} site(s), "
        f"leads {args.leads}h, splits {args.splits}"
    )
    print(
        "Operational lead per horizon: "
        + ", ".join(
            f"{label}->{operational[steps]}h"
            for steps, label in zip(HORIZON_STEPS, HORIZON_LABELS)
        )
        + "\n"
    )

    started = time.time()
    rows_written = 0

    for site in args.sites:
        site_start = time.time()
        frame = load_site(site)
        index = frame.index
        ghi = frame["GHI"].to_numpy(dtype=float)
        envelope = clearsky_ghi(index, site).to_numpy()
        nsrdb_envelope = frame["Clearsky GHI"].to_numpy(dtype=float)
        observed_csi = clear_sky_index(ghi, envelope)
        splits = split_label(index)

        predictions, fits = build_predictions(
            site, args.leads, index, envelope, observed_csi
        )
        if not predictions:
            print(f"  {site:<16} no usable forecast fields, skipped")
            continue

        summary = ", ".join(
            f"{lead}h a={f['a']:.3f} b={f['b']:.3f}"
            for lead, f in sorted(fits.items())
        )
        print(f"  {site:<16} fitted {summary}")

        for steps, label in zip(HORIZON_STEPS, HORIZON_LABELS):
            reference = {
                PRIMARY_FS_REFERENCE: smart_persistence(
                    ghi, nsrdb_envelope, steps
                ),
                "naive_persistence": naive_persistence(ghi, steps),
            }

            for (name, lead), values in predictions.items():
                for split in args.splits:
                    # Requiring the references to be finite as well keeps
                    # Forecast Skill defined on exactly the samples it is
                    # reported for. Over the scored splits this removes
                    # nothing -- the persistence gaps sit at the very
                    # start of the record, in a training year.
                    finite = (
                        (splits == split)
                        & np.isfinite(values)
                        & np.isfinite(reference[PRIMARY_FS_REFERENCE])
                        & np.isfinite(reference["naive_persistence"])
                    )
                    if finite.sum() < 30:
                        continue

                    scored = score_predictions(
                        ghi[finite],
                        values[finite],
                        envelope[finite],
                        index[finite],
                        reference_smart=reference[PRIMARY_FS_REFERENCE][
                            finite
                        ],
                        reference_naive=reference["naive_persistence"][finite],
                    )
                    append_results(
                        scored,
                        out_path,
                        meta=meta[name],
                        site=site,
                        horizon_label=label,
                        horizon_steps=steps,
                        # The lead is the only thing distinguishing two
                        # rows of the same model at the same horizon, so
                        # it travels in the track column.
                        track=f"jma_lead{lead}h",
                        feature_set="nwp",
                        target="ghi",
                        split=split,
                    )
                    rows_written += len(scored)

        print(f"  {site:<16} scored in {_duration(time.time() - site_start)}")

    print(
        f"\nWrote {rows_written} rows to {out_path}  "
        f"({_duration(time.time() - started)} total)"
    )
    _print_incumbent_table(out_path, args.sites[0], operational)
    return 0


def _print_incumbent_table(path, site: str, operational) -> None:
    """Print the bar every later model has to clear, at its own lead."""
    from solarfc.results import load_results

    frame = load_results(path)

    print(f"\nOperational NWP incumbent, {site}, test split, daytime only.")
    print("Each horizon at the shortest lead a real forecast could use.\n")
    print(
        f"{'horizon':>8} {'lead':>5} {'MOS MAE':>9} {'MOS RMSE':>9} "
        f"{'cloud MAE':>10} {'KC MAE':>8}"
    )

    for steps, label in zip(HORIZON_STEPS, HORIZON_LABELS):
        lead = operational[steps]
        subset = frame[
            (frame["site"] == site)
            & (frame["split"] == "test")
            & (frame["stratum"] == "all")
            & (frame["horizon_label"] == label)
            & (frame["track"] == f"jma_lead{lead}h")
        ].set_index("model")
        if subset.empty:
            continue

        def cell(model: str, column: str) -> str:
            if model not in subset.index:
                return f"{'-':>9}"
            return f"{subset.loc[model, column]:>9.1f}"

        print(
            f"{label:>8} {lead:>4}h {cell('nwp_jma_mos', 'mae')} "
            f"{cell('nwp_jma_mos', 'rmse')} "
            f"{cell('nwp_jma_cloud', 'mae')} "
            f"{cell('nwp_jma_cloud_kc', 'mae').strip():>8}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
