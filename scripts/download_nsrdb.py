"""
Download NSRDB Himawari PSM v3 data (2016-2020) for all 7 SEA sites.
10-minute interval, all relevant attributes.

Usage:
    pip install requests python-dotenv
    python download_nsrdb.py
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NREL_API_KEY")
EMAIL = os.getenv("NREL_EMAIL")  # Add your email to .env

if not API_KEY:
    raise ValueError("NREL_API_KEY not found in .env")
if not EMAIL:
    raise ValueError("NREL_EMAIL not found in .env — add your email to .env")

BASE_URL = f"https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-download.csv?api_key={API_KEY}"

SITES = {
    "kuala_lumpur": (3.139, 101.687),
    "penang": (5.414, 100.330),
    "kota_kinabalu": (5.980, 116.073),
    "ho_chi_minh": (10.823, 106.630),
    "bangkok": (13.754, 100.501),
    "jakarta": (-6.208, 106.846),
    "manila": (14.599, 120.984),
}

YEARS = [2016, 2017, 2018, 2019, 2020]

ATTRIBUTES = ",".join(
    [
        "ghi",
        "dni",
        "dhi",
        "clearsky_ghi",
        "clearsky_dni",
        "clearsky_dhi",
        "air_temperature",
        "dew_point",
        "relative_humidity",
        "surface_pressure",
        "wind_speed",
        "wind_direction",
        "total_precipitable_water",
        "surface_albedo",
        "cloud_type",
        "fill_flag",
        "solar_zenith_angle",
        "aod",
        "alpha",
        "ozone",
        "asymmetry",
    ]
)

HEADERS = {
    "content-type": "application/x-www-form-urlencoded",
    "cache-control": "no-cache",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "nsrdb")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_site_year(site_name, lat, lon, year):
    """Download a single site-year combination."""
    output_file = os.path.join(OUTPUT_DIR, f"{site_name}_{year}.csv")
    if os.path.exists(output_file):
        print(f"  [SKIP] {output_file} already exists")
        return True

    wkt = f"POINT({lon} {lat})"
    payload = (
        f"wkt={wkt}"
        f"&attributes={ATTRIBUTES}"
        f"&names={year}"
        f"&utc=true"
        f"&leap_day=true"
        f"&interval=10"
        f"&email={EMAIL}"
    )

    print(f"  [POST] {site_name} {year} ...")
    response = requests.post(BASE_URL, data=payload, headers=HEADERS)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        size_mb = len(response.content) / (1024 * 1024)
        print(f"  [OK]   {output_file} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  [FAIL] {site_name} {year}: HTTP {response.status_code}")
        print(f"         {response.text[:300]}")
        return False


def main():
    total = len(SITES) * len(YEARS)
    done = 0

    for site_name, (lat, lon) in SITES.items():
        print(f"\n=== {site_name.upper()} ({lat}, {lon}) ===")
        for year in YEARS:
            done += 1
            print(f"[{done}/{total}]", end="")
            download_site_year(site_name, lat, lon, year)
            # Rate limit: max 1 request per 2 seconds for CSV
            time.sleep(2)

    print(f"\nDone. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
