"""Pull TIGGE ensemble forecasts for the predictability study.

Purpose: obtain the **predictability error growth** (PEG) that Yang's
solar-predictability framework needs. PEG is the divergence between a
control forecast and its perturbed siblings as lead time grows, and it
supplies the *lower* bound on mean square error -- the smallest error any
forecast could attain in this atmosphere. The correlogram-based upper
bound comes from the clear-sky index and needs no download at all.

Yang mapped this over seven sites in the United States. Nobody has done
it for equatorial Southeast Asia, and nobody has stratified it by monsoon
phase, which is why the full year is pulled rather than a sample.

What the archive dictates
-------------------------
The ECMWF Data Store form has **no member selector**, so asking for
``perturbed_forecast`` returns all fifty members whether or not they are
wanted. Subsetting can only happen after download. That is why the
per-request cost is what it is.

Measured before writing this: one day, two lead times, fifty members is
5.25 MB and 273 seconds of server time. Almost all of that is the server
building the job, not the transfer, so requests are batched **by month**.
Twelve months times two forecast types is 24 jobs rather than 730.

Lead times run 6-hourly to 48 h, then daily to 168 h. The dense part
covers every regulated horizon and the steep early error growth; the
sparse tail locates where predictability saturates. Lead 0 is the
analysis, kept as the reference the perturbations are measured against.

``ssr`` is surface *net* solar radiation accumulated from forecast start,
in J m-2, so ``extract_sites`` differences successive lead times and
divides by the interval to recover a mean flux.

It stays **net**, not global. ``ssr = GHI * (1 - albedo)``, and the
conversion is deliberately left to the analysis step rather than baked
in here: NSRDB carries a per-timestep ``surface_albedo`` that is better
than any constant this script could assume, and the predictability
statistic is a control-versus-perturbed spread in which the albedo
factor very nearly cancels anyway.

The grid is a **reduced Gaussian** one, not a regular latitude-longitude
mesh: latitude and longitude arrive as flat arrays over a ``values``
dimension with a different point count per row. Nearest-neighbour
extraction therefore searches those arrays directly rather than indexing
a 2-D grid. Spacing is about 0.14 degrees, so the nearest point to Kuala
Lumpur is 4.5 km away -- considerably closer than ERA5's 31 km.

Credentials
-----------
Deliberately **not** ``~/.cdsapirc``. That file belongs to the Copernicus
Climate Data Store and drives the ERA5 pipeline; the ECMWF Data Store
wants the same filename with a different url and key, and overwriting it
would silently break ERA5 downloads. The token lives in ``.env`` as
``ECDS_TOKEN`` and is passed to the client explicitly.

Licence
-------
The TIGGE licence splits by producing centre. The non-commercial half
(BoM, CMA, CPTEC, IMD, JMA, MF, NCMRWF) is CC BY-NC 4.0, but **ECMWF's
own contribution is CC BY 4.0** -- and ``origin`` here is ``ecmwf``.

So this data is redistributable with attribution, on the same terms as
ERA5, and it does not force the benchmark deposit to a non-commercial
licence. Published work must acknowledge TIGGE. Do not widen ``origin``
to another centre without re-checking, since that would pull NC data in
and change the deposit's licence.

Accept the licence at
https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download#manage-licences
before the first run or every request returns 403.

Run:
    python scripts/pull_tigge_ensemble.py
    python scripts/pull_tigge_ensemble.py --months 1 2 3
    python scripts/pull_tigge_ensemble.py --extract-only

Roughly 13.6 GB and 5-8 hours, almost all of it queue. Measured: a
perturbed month is ~1109 MB and 25-40 min, a control month ~21 MB and
90 s. ECDS runs one job at a time per user, so parallel processes only
queue behind each other and buy nothing. Resumable: a month
already on disk is skipped, so an interrupted run continues. Raw GRIB
goes to ~/tigge_raw, outside the iCloud-synced repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from solarfc.config import PROCESSED_DIR, SITE_KEYS, SITES_BY_KEY

DATASET = "tigge-forecasts"

#: Raw GRIB lives **outside** the repository on purpose. The repository
#: sits in iCloud Drive, so anything written under it is uploaded to the
#: user's iCloud quota -- and this download is roughly 13 GB. Only the
#: extracted per-site Parquet, which is small, belongs in the repo.
RAW_DIR = Path.home() / "tigge_raw"
OUT_DIR = PROCESSED_DIR / "tigge"

#: The seven-site bounding box, as [North, West, South, East].
AREA = [16, 98, -8, 122]

#: Lead times in hours, deliberately uneven.
#:
#: Six-hourly through 48 h resolves the steep early growth of forecast
#: error, and covers every horizon the grid code regulates. Beyond that
#: the curve is smooth, so daily sampling to seven days locates the
#: saturation point -- the lead time past which predictability stops
#: decaying -- without paying for detail that carries no information.
#:
#: Two lead times would have given Yang's bound only at 24 h and 48 h.
#: This gives the whole curve, which is what supports a statement about
#: an equatorial predictability *horizon* rather than two isolated
#: numbers.
LEAD_HOURS = [
    "0",
    "6",
    "12",
    "18",
    "24",
    "30",
    "36",
    "42",
    "48",
    "72",
    "96",
    "120",
    "144",
    "168",
]

FORECAST_TYPES = ("control_forecast", "perturbed_forecast")

#: Single initialisation per day. TIGGE also carries 06/12/18, but a
#: second cycle would double the download to sharpen a statistic that is
#: already averaged over 365 days.
INIT_TIME = "00:00"

YEAR = "2020"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=list(range(1, 13)),
        help="calendar months to pull",
    )
    p.add_argument("--year", default=YEAR)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="skip downloading, rebuild the Parquet cache from GRIB on disk",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help=(
            "fetch GRIB and stop. Use this when running several months in "
            "parallel: every process would otherwise extract and write the "
            "same Parquet, racing each other and each seeing only its own "
            "months. Run one --extract-only pass afterwards instead."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true", help="re-download existing months"
    )
    return p.parse_args(argv)


def client():
    """A Data Store client authenticated from .env, never ~/.cdsapirc."""
    try:
        import cdsapi
    except ImportError:  # pragma: no cover
        raise SystemExit("pip install 'cdsapi>=0.7.7'")

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    url = os.environ.get("ECDS_URL", "https://ecds.ecmwf.int/api")
    key = os.environ.get("ECDS_TOKEN")
    if not key:
        raise SystemExit(
            "ECDS_TOKEN missing from .env. Get one from "
            "https://ecds.ecmwf.int/how-to-api"
        )
    return cdsapi.Client(url=url, key=key)


def month_path(raw_dir: Path, year: str, month: int, kind: str) -> Path:
    return raw_dir / f"tigge_{year}{month:02d}_{kind}.grib"


def download_month(api, year: str, month: int, kind: str, target: Path):
    """One request covering a whole month for one forecast type.

    Downloads to a ``.part`` file and renames only once the transfer has
    finished. Resumption tests whether the final name exists, so a
    half-written file left behind by an interrupted run would otherwise
    be treated as complete and silently feed truncated data into the
    extract -- which is exactly what happened to one month during
    development, at 303 MB against a normal 1109 MB.
    """
    days = pd.Period(f"{year}-{month:02d}").days_in_month
    request = {
        "origin": "ecmwf",
        "level_type": "single_level",
        "variable": ["surface_net_solar_radiation"],
        "forecast_type": kind,
        "year": [year],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, days + 1)],
        "time": [INIT_TIME],
        "leadtime_hour": LEAD_HOURS,
        "area": AREA,
        "data_format": "grib",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    api.retrieve(DATASET, request, str(partial))
    partial.replace(target)


def nearest_index(lat: np.ndarray, lon: np.ndarray, site) -> tuple[int, float]:
    """Index of the closest grid point, and its distance in km.

    The reduced Gaussian grid has no row structure to exploit, so this is
    a direct search over the flat coordinate arrays. Longitude is scaled
    by cos(latitude) so the comparison is a real distance rather than a
    degree difference that means different things on each axis.
    """
    dlat = lat - site.latitude
    dlon = (lon - site.longitude) * np.cos(np.radians(site.latitude))
    d = np.hypot(dlat, dlon)
    i = int(np.argmin(d))
    return i, float(d[i] * 111.0)


def extract_sites(path: Path) -> pd.DataFrame:
    """Site time series from one monthly GRIB, in W/m^2 per lead interval.

    Returns long format: one row per (valid time, site, member, lead),
    which keeps the control and the fifty perturbed members in the same
    frame and lets the spread be computed by grouping.
    """
    import warnings

    import xarray as xr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(path, engine="cfgrib")

    lat = ds.latitude.values
    lon = ds.longitude.values

    parts = []
    for site_key in SITE_KEYS:
        site = SITES_BY_KEY[site_key]
        i, distance_km = nearest_index(lat, lon, site)

        # Selecting the grid point with xarray keeps whatever dimensions
        # the file happens to carry. A monthly download has `time` as a
        # dimension where a single-day one has it as a scalar, and only
        # the perturbed files have `number` -- indexing the raw numpy
        # array by position gets that wrong the moment the shape changes.
        frame = (
            ds.ssr.isel(values=i).to_dataframe(name="ssr_accum").reset_index()
        )
        if "number" not in frame.columns:
            frame["number"] = 0

        frame["site"] = site_key
        frame["grid_distance_km"] = distance_km
        parts.append(frame)

    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={"number": "member", "time": "init_time"})
    out["lead_hours"] = (
        pd.to_timedelta(out["step"]).dt.total_seconds() // 3600
    ).astype(int)

    # ssr accumulates from forecast start, so the value at each lead is
    # the total since step 0. Differencing successive leads gives the
    # energy over that interval, and dividing by the interval gives a
    # mean flux comparable to an instantaneous series. The lead spacing
    # is deliberately uneven, so the divisor has to come from the actual
    # step difference rather than a constant.
    out = out.sort_values(["site", "member", "init_time", "lead_hours"])
    group = out.groupby(["site", "member", "init_time"], sort=False)
    joules = group["ssr_accum"].diff()
    seconds = group["lead_hours"].diff() * 3600.0

    out["interval_hours"] = (seconds / 3600.0).astype("Float64")
    out["ssr_w_m2"] = joules / seconds

    # The first lead of each forecast has no preceding step to difference
    # against, so it carries no interval flux.
    out = out[out["ssr_w_m2"].notna()].copy()

    out["init_time"] = pd.to_datetime(out["init_time"], utc=True)
    out["valid_time"] = out["init_time"] + pd.to_timedelta(
        out["lead_hours"], unit="h"
    )
    keep = [
        "site",
        "member",
        "init_time",
        "valid_time",
        "lead_hours",
        "interval_hours",
        "ssr_w_m2",
        "grid_distance_km",
    ]
    return out[keep].reset_index(drop=True)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def main(argv=None) -> int:
    args = parse_args(argv)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(m, k) for m in args.months for k in FORECAST_TYPES]
    started = time.time()

    if not args.extract_only:
        api = client()
        print(
            f"TIGGE {args.year}: {len(args.months)} months x "
            f"{len(FORECAST_TYPES)} forecast types = {len(jobs)} requests"
        )
        print(f"  leads {LEAD_HOURS}h, area {AREA}, writing to {args.raw_dir}")
        print("  expect roughly 13.6 GB and 5-8 hours, mostly queue\n")

        for n, (month, kind) in enumerate(jobs, start=1):
            target = month_path(args.raw_dir, args.year, month, kind)
            if target.exists() and not args.overwrite:
                print(
                    f"  [{n:>2}/{len(jobs)}] {args.year}-{month:02d} "
                    f"{kind:<19} already on disk, skipped"
                )
                continue

            t0 = time.time()
            try:
                download_month(api, args.year, month, kind, target)
            except Exception as exc:
                print(
                    f"  [{n:>2}/{len(jobs)}] {args.year}-{month:02d} "
                    f"{kind:<19} FAILED: {exc}",
                    file=sys.stderr,
                )
                continue
            size_mb = target.stat().st_size / 1e6
            print(
                f"  [{n:>2}/{len(jobs)}] {args.year}-{month:02d} "
                f"{kind:<19} {size_mb:6.1f} MB  {_duration(time.time() - t0)}"
                f"   (elapsed {_duration(time.time() - started)})",
                flush=True,
            )

    if args.download_only:
        print(
            f"\ndownloads finished in {_duration(time.time() - started)}. "
            "Run with --extract-only once every month is on disk."
        )
        return 0

    print("\nextracting sites")
    frames, manifest_files = [], []
    for month, kind in jobs:
        path = month_path(args.raw_dir, args.year, month, kind)
        if not path.exists():
            continue
        part = extract_sites(path)
        part["forecast_type"] = kind
        frames.append(part)
        manifest_files.append(
            {
                "file": path.name,
                "month": month,
                "forecast_type": kind,
                "bytes": path.stat().st_size,
                "rows_extracted": len(part),
                "sha256": sha256(path),
            }
        )
        print(f"  {path.name:<44} {len(part):>8,} rows")

    if not frames:
        print("nothing extracted -- no GRIB files found", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["site", "forecast_type", "member", "valid_time"]
    )
    out = args.out_dir / f"tigge_ensemble_{args.year}.parquet"
    combined.to_parquet(out, index=False)

    distances = combined.groupby("site").grid_distance_km.first()
    print(f"\nwrote {out}  ({len(combined):,} rows)")
    print("\nnearest grid point per site:")
    for site_key, km in distances.items():
        print(f"  {site_key:<16} {km:5.1f} km")

    manifest = {
        "dataset": DATASET,
        "endpoint": "https://ecds.ecmwf.int/api",
        "origin": "ecmwf",
        "year": args.year,
        "lead_hours": LEAD_HOURS,
        "init_time": INIT_TIME,
        "area_north_west_south_east": AREA,
        "variable": "surface_net_solar_radiation (ssr), J m-2 accumulated",
        "purpose": (
            "Predictability error growth (control vs perturbed) for the "
            "lower bound on mean square error over equatorial SEA."
        ),
        "licence": (
            "CC BY 4.0 for ECMWF-origin TIGGE data (the non-commercial "
            "CC BY-NC 4.0 half of TIGGE covers other centres, not this "
            "one). Redistributable with attribution, compatible with the "
            "benchmark deposit. Published work must acknowledge TIGGE."
        ),
        "grid": (
            "Reduced Gaussian, ~0.14 deg. Nearest-neighbour extraction "
            "searches the flat coordinate arrays directly."
        ),
        "files": manifest_files,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'manifest.json'}")
    print(f"total {_duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
