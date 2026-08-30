"""Extract, upsample and cache ERA5 site series to Parquet.

Reading the 60 monthly ERA5 archives is slow — each is a ~100 MB zip
holding two NetCDF members on a 97x97 grid. Only seven gridpoints are
ever used, so this runs once and caches the result.

The loop is deliberately month-outer / site-inner: reading each archive
once and extracting all seven sites costs 60 file reads, where a
site-outer loop would cost 420.

Run:
    python scripts/build_era5_cache.py python
    scripts/build_era5_cache.py --years 2016 2017 --sites kuala_lumpur

Output:
    data/processed/era5/{site}_10min.parquet   upsampled, derived features
    data/processed/era5/manifest.json          checksums + provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from solarfc import era5
from solarfc.config import (
    ALL_YEARS,
    ERA5_DIR,
    PROCESSED_DIR,
    SITE_KEYS,
    SITES_BY_KEY,
)

OUT_DIR = PROCESSED_DIR / "era5"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", type=int, nargs="+", default=list(ALL_YEARS))
    p.add_argument("--sites", nargs="+", default=list(SITE_KEYS))
    p.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument(
        "--cloud-method",
        choices=("linear", "ffill"),
        default="linear",
        help="tcc upsampling. Default linear: ERA5 tcc is an instantaneous "
        "field, so linear is the correct reconstruction of what is stored.",
    )
    p.add_argument("--data-dir", type=Path, default=ERA5_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args(argv)


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

    months = [(y, m) for y in sorted(args.years) for m in sorted(args.months)]
    missing = [
        f"era5_sea_{y}_{m:02d}.nc"
        for y, m in months
        if not (args.data_dir / f"era5_sea_{y}_{m:02d}.nc").exists()
    ]
    if missing:
        print(
            f"missing {len(missing)} ERA5 files, first few: {missing[:5]}",
            file=sys.stderr,
        )
        return 1

    print(
        f"reading {len(months)} monthly archives for {len(args.sites)} sites"
    )
    per_site: dict[str, list[pd.DataFrame]] = {s: [] for s in args.sites}
    attrs: dict[str, dict] = {}

    t0 = time.time()
    for i, (year, month) in enumerate(months, start=1):
        path = args.data_dir / f"era5_sea_{year}_{month:02d}.nc"
        with era5.open_month(path) as ds:
            for site in args.sites:
                part = era5.extract_site(ds, site)
                attrs[site] = part.attrs
                per_site[site].append(part)
        elapsed = time.time() - t0
        rate = elapsed / i
        print(
            f"  [{i:>2}/{len(months)}] {path.name}  "
            f"{elapsed:5.1f}s elapsed, ~{rate*(len(months)-i):5.1f}s left",
            flush=True,
        )

    print("\nupsampling to 10-minute and deriving features")
    manifest_files = []
    for site in args.sites:
        hourly = pd.concat(per_site[site]).sort_index()

        dupes = int(hourly.index.duplicated().sum())
        if dupes:
            print(f"  {site}: {dupes} duplicate timestamps", file=sys.stderr)
            return 1

        hourly.attrs.update(attrs[site])
        ten_min = era5.upsample_to_10min(
            hourly, cloud_method=args.cloud_method
        )
        derived = era5.derive_features(ten_min)

        # Keep only the model-facing columns; the raw ERA5 names are
        # redundant once units are aligned, and carrying both invites
        # using the wrong one.
        keep = [c for c in derived.columns if c.startswith("era5_")]
        out = derived[keep]

        path = args.out_dir / f"{site}_10min.parquet"
        out.to_parquet(path, index=True)

        offset = attrs[site]["offset_km"]
        print(
            f"  {site:<16} {len(out):>7,} rows  "
            f"{out.index.min().date()} -> {out.index.max().date()}  "
            f"gridpoint {offset:4.1f} km away  "
            f"{path.stat().st_size/1e6:5.1f} MB"
        )
        manifest_files.append(
            {
                "site": site,
                "file": path.name,
                "rows": len(out),
                "columns": list(out.columns),
                "start": str(out.index.min()),
                "end": str(out.index.max()),
                "grid_lat": attrs[site]["grid_lat"],
                "grid_lon": attrs[site]["grid_lon"],
                "offset_km": round(offset, 2),
                "sha256": sha256(path),
            }
        )

    manifest = {
        "source": "ERA5 reanalysis, Copernicus CDS",
        "spatial_method": "nearest gridpoint",
        "spatial_method_rationale": (
            "At 0.25 degrees a bilinear stencil around a coastal site blends "
            "land and sea cells, which have different diurnal temperature "
            "range and cloud regimes. Five of the seven sites are coastal, so "
            "nearest keeps each site on one physically coherent cell."
        ),
        "temporal_method": {
            "instantaneous": "linear interpolation (tcc, t2m, d2m, msl, u10, v10, tcwv)",
            "accumulated": (
                "shifted back one hour to the start of the window they "
                "describe, converted to a mean rate, then held constant "
                "(tp, ssrd). Verified against NSRDB GHI: cross-correlation "
                "peaks at zero lag with the shift and at -60 min without it."
            ),
        },
        "cloud_method": args.cloud_method,
        "step_minutes": 10,
        "years": sorted(args.years),
        "files": manifest_files,
    }

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path}")
    print(f"total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
