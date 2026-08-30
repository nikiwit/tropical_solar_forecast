"""Pull archived JMA GSM forecasts from Open-Meteo's Previous Runs API.

Purpose: measure how real NWP forecast error grows with lead time over
the seven study sites, so the realistic known-future track is calibrated
from data rather than from published verification figures (which are
overwhelmingly European or CONUS, where convective cloud behaves
differently).

The API serves each variable at fixed lead offsets. ``_previous_day1``
is what was forecast 24 hours before the valid time, ``day2`` 48 hours
before, and so on. Comparing those against ERA5 gives real error
statistics per variable per lead time.

What is actually available, probed rather than taken from the docs:

* JMA GSM carries cloud cover, 2 m temperature, relative humidity, dew point
  and precipitation back to 2018 -- the five fields the degradation
  model perturbs.
* Offsets day0, day1, day2 and day3 are populated. day5 and day7 are not.
* JMA GSM has **no** shortwave radiation at any date. Archived forecast GHI
  begins in 2024 for every model, and NSRDB ends in 2020, so an
  operational NWP GHI baseline on the test year is impossible. That
  comparison moves to the real-world deployment demo instead.

Run:
    python scripts/pull_jma_forecasts.py python
    scripts/pull_jma_forecasts.py --years 2020 --sites kuala_lumpur

Output:
    data/processed/jma/{site}_forecasts.parquet
    data/processed/jma/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from solarfc.config import PROCESSED_DIR, SITE_KEYS, SITES_BY_KEY

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "jma_gsm"
OUT_DIR = PROCESSED_DIR / "jma"

#: API variable -> (our column stem, conversion to ERA5 units).
#:
#: Units must match the ERA5 columns exactly, or the measured "error"
#: would be a unit mismatch rather than forecast error. JMA reports
#: cloud cover as a percentage; ERA5 uses a 0-1 fraction.
VARIABLES: dict[str, tuple[str, float]] = {
    "cloud_cover": ("era5_cloud_cover", 0.01),
    "temperature_2m": ("era5_temp_c", 1.0),
    "relative_humidity_2m": ("era5_relative_humidity", 1.0),
    "dew_point_2m": ("era5_dewpoint_c", 1.0),
    "precipitation": ("era5_precip_mm_h", 1.0),
}

#: Populated lead offsets. day5/day7 return nulls for this model.
LEAD_DAYS: tuple[int, ...] = (0, 1, 2, 3)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020])
    p.add_argument("--sites", nargs="+", default=list(SITE_KEYS))
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Seconds between requests. The API is free; do not hammer it.",
    )
    p.add_argument("--timeout", type=int, default=300)
    return p.parse_args(argv)


def _request_names() -> list[str]:
    names = []
    for var in VARIABLES:
        for day in LEAD_DAYS:
            names.append(var if day == 0 else f"{var}_previous_day{day}")
    return names


def fetch_site_year(site_key: str, year: int, timeout: int) -> pd.DataFrame:
    """One request covering a full year for one site."""
    site = SITES_BY_KEY[site_key]
    url = (
        f"{API}?latitude={site.latitude}&longitude={site.longitude}"
        f"&start_date={year}-01-01&end_date={year}-12-31"
        f"&hourly={','.join(_request_names())}"
        f"&models={MODEL}&timezone=UTC"
    )

    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)

    hourly = payload["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), tz="UTC")
    index.name = "timestamp"

    frame = pd.DataFrame(index=index)
    for var, (stem, scale) in VARIABLES.items():
        for day in LEAD_DAYS:
            key = var if day == 0 else f"{var}_previous_day{day}"
            if key not in hourly:
                continue
            values = pd.Series(hourly[key], index=index, dtype="float64")
            frame[f"{stem}_lead{day * 24}h"] = values * scale

    frame.attrs["grid_lat"] = payload["latitude"]
    frame.attrs["grid_lon"] = payload["longitude"]
    return frame


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)

    unknown = [s for s in args.sites if s not in SITES_BY_KEY]
    if unknown:
        print(f"unknown sites: {unknown}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, y) for s in args.sites for y in sorted(args.years)]

    print(
        f"pulling {MODEL}: {len(args.sites)} sites x {len(args.years)} years "
        f"= {len(jobs)} requests, leads {[d * 24 for d in LEAD_DAYS]}h"
    )

    per_site: dict[str, list[pd.DataFrame]] = {s: [] for s in args.sites}
    attrs: dict[str, dict] = {}
    start = time.time()

    for i, (site_key, year) in enumerate(jobs, start=1):
        for attempt in (1, 2, 3):
            try:
                part = fetch_site_year(site_key, year, args.timeout)
                break
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                if attempt == 3:
                    print(
                        f"\n  {site_key} {year}: failed after 3 attempts: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                backoff = 5 * attempt
                print(
                    f"    retry {attempt} for {site_key} {year} in {backoff}s "
                    f"({type(exc).__name__})",
                    flush=True,
                )
                time.sleep(backoff)

        per_site[site_key].append(part)
        attrs[site_key] = dict(part.attrs)

        elapsed = time.time() - start
        print(
            f"  [{i:>2}/{len(jobs)}] {site_key:<16} {year}  "
            f"{len(part):>5} rows  {elapsed:5.1f}s elapsed",
            flush=True,
        )

        if i < len(jobs):
            time.sleep(args.pause)

    print("\nwriting per-site caches")
    manifest_files = []
    for site_key in args.sites:
        frame = pd.concat(per_site[site_key]).sort_index()

        dupes = int(frame.index.duplicated().sum())
        if dupes:
            print(
                f"  {site_key}: {dupes} duplicate timestamps", file=sys.stderr
            )
            return 1

        path = args.out_dir / f"{site_key}_forecasts.parquet"
        frame.to_parquet(path, index=True)

        coverage = {
            c: round(float(frame[c].notna().mean()), 4) for c in frame.columns
        }
        worst = min(coverage.values()) if coverage else 0.0
        print(
            f"  {site_key:<16} {len(frame):>6,} rows  "
            f"{frame.index.min().date()} -> {frame.index.max().date()}  "
            f"min coverage {worst:.1%}"
        )

        manifest_files.append(
            {
                "site": site_key,
                "file": path.name,
                "rows": len(frame),
                "start": str(frame.index.min()),
                "end": str(frame.index.max()),
                "grid_lat": attrs[site_key].get("grid_lat"),
                "grid_lon": attrs[site_key].get("grid_lon"),
                "coverage": coverage,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "source": "Open-Meteo Previous Runs API",
        "model": MODEL,
        "endpoint": API,
        "lead_hours": [d * 24 for d in LEAD_DAYS],
        "variables": {k: v[0] for k, v in VARIABLES.items()},
        "unit_conversions": {
            "cloud_cover": "percent -> fraction (x0.01), to match ERA5 tcc",
        },
        "purpose": (
            "Measure real NWP forecast error by lead time over tropical SEA, "
            "to calibrate the realistic known-future track from data rather "
            "than from European/CONUS verification literature."
        ),
        "known_limitation": (
            "JMA GSM carries no shortwave radiation at any date, and archived "
            "forecast GHI begins in 2024 for every model while NSRDB ends in "
            "2020. An operational NWP GHI baseline on the test year is "
            "therefore impossible; that comparison moves to the real-world "
            "real-world demo against Solcast actuals."
        ),
        "years": sorted(args.years),
        "files": manifest_files,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out_dir / 'manifest.json'}")
    print(f"total {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
