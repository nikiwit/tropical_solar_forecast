"""Fit Linke turbidity per site and write the calibration artefact.

pvlib's default Linke turbidity climatology underestimates clear-sky GHI by
4.4-7.6% at all seven sites, which puts a systematic multiplicative error into
every clear-sky index in the project. This fits one turbidity value per site
against its own observed clear-sky periods, using only the training years.

Run once. The output is committed -- it is a result, not raw data, and the rest
of the pipeline must be reproducible without re-running the fit.

    python scripts/fit_turbidity.py
    python scripts/fit_turbidity.py --sites kuala_lumpur penang

Takes roughly 4-8 minutes for all seven sites.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

from solarfc.clearsky import fit_linke_turbidity, save_turbidity, TURBIDITY_PATH
from solarfc.config import SITE_KEYS, TRAIN_YEARS
from solarfc.data import load_site


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(SITE_KEYS))
    parser.add_argument(
        "--out", default=None, help=f"output path (default {TURBIDITY_PATH})"
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    print(f"Fitting Linke turbidity on training years {TRAIN_YEARS}")
    print(f"{len(args.sites)} site(s); detect-then-fit, up to 5 iterations each\n")

    def progress(site, iteration, value, n_clear, rmse):
        print(
            f"    iter {iteration}: TL={value:6.3f}  clear={n_clear:>7,}  "
            f"RMSE={rmse:6.2f} W/m2",
            flush=True,
        )

    results: dict[str, dict] = {}
    started = time.time()

    for position, site in enumerate(args.sites, start=1):
        site_start = time.time()
        elapsed = time.time() - started
        eta = (elapsed / (position - 1) * (len(args.sites) - position + 1)) if position > 1 else None
        header = f"[{position}/{len(args.sites)}] {site}"
        if eta is not None:
            header += f"   (elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)})"
        print(header, flush=True)

        frame = load_site(site)
        outcome = fit_linke_turbidity(frame["GHI"], site, progress=progress)
        results[site] = outcome

        shift = 100.0 * (outcome["turbidity"] / outcome["default_turbidity"] - 1.0)
        flag = "" if outcome["converged"] else "  [did not converge]"
        print(
            f"  -> TL {outcome['default_turbidity']:.2f} (pvlib) "
            f"-> {outcome['turbidity']:.3f} fitted  ({shift:+.1f}%)  "
            f"RMSE {outcome['rmse_clear']:.2f} W/m2 on {outcome['n_clear']:,} clear "
            f"samples in {_format_duration(time.time() - site_start)}{flag}\n",
            flush=True,
        )

    path = save_turbidity(results, args.out)
    print(f"Wrote {path}  ({_format_duration(time.time() - started)} total)")
    print("\nCommit this file -- it is a fitted result the pipeline depends on.")

    not_converged = [s for s, r in results.items() if not r["converged"]]
    if not_converged:
        print(f"\nWARNING: did not converge for {not_converged}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
