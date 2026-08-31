"""
Download Solcast historical radiation + weather data for
spot-validation.

Used to cross-validate NSRDB Himawari data at key sites. The free
researcher account has 50 API requests total — each request covers up to
31 days.

Default config (KL + Penang, 2019): 24 requests, well within the 50
limit. Extend YEARS or SITES only if you have remaining quota.

Prerequisites:
    pip install solcast python-dotenv
    Add SOLCAST_API_KEY=your_key to .env

Usage:
    python download_solcast.py

API docs: https://docs.solcast.com.au/
SDK docs: https://solcast.github.io/solcast-api-python-sdk/
"""

import os
import calendar
from datetime import datetime
from dotenv import load_dotenv
from solcast import historic

load_dotenv()

api_key = os.getenv("SOLCAST_API_KEY")
if not api_key:
    raise ValueError("SOLCAST_API_KEY not found in .env")

os.environ["SOLCAST_API_KEY"] = api_key  # SDK reads from env

# --- Config --- Default: 2 sites × 1 year × 12 months = 24 requests
# (leaves 26 in reserve) To extend: add more years or sites, but keep
# total requests under 50.
SITES = {
    "kuala_lumpur": (3.139, 101.687),
    "penang": (5.414, 100.330),
}

YEARS = [2019]  # extend to e.g. [2018, 2019, 2020] only if quota allows

PERIOD = "PT30M"  # 30-minute intervals — matches Solcast's native resolution

OUTPUT_PARAMETERS = ",".join(
    [
        "ghi",
        "clearsky_ghi",
        "dni",
        "dhi",
        "cloud_opacity",
        "air_temp",
        "relative_humidity",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitable_water",
    ]
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "solcast")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Request counter ---
requests_made = 0
QUOTA = 50


def download_month(site_name, lat, lon, year, month):
    global requests_made

    output_file = os.path.join(
        OUTPUT_DIR, f"{site_name}_{year}_{month:02d}.csv"
    )
    if os.path.exists(output_file):
        print(f"  [SKIP] {output_file} already exists")
        return True

    if requests_made >= QUOTA:
        print(f"  [STOP] Quota limit of {QUOTA} requests reached — stopping.")
        return False

    # ISO 8601 start/end for the calendar month
    start = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(
        f"  [GET] {site_name} {year}-{month:02d} ({start_str} → {end_str}) ..."
    )
    try:
        res = historic.radiation_and_weather(
            latitude=lat,
            longitude=lon,
            output_parameters=OUTPUT_PARAMETERS,
            start=start_str,
            end=end_str,
            period=PERIOD,
        )
        requests_made += 1

        df = res.to_pandas()
        df.to_csv(output_file)
        size_kb = os.path.getsize(output_file) / 1024
        print(
            f"  [OK]   {output_file} ({size_kb:.0f} KB) [{requests_made}/{QUOTA} requests used]"
        )
        return True

    except Exception as e:
        print(f"  [FAIL] {site_name} {year}-{month:02d}: {e}")
        return False


def main():
    total = len(SITES) * len(YEARS) * 12
    done = 0

    print(f"Quota: {QUOTA} requests. Planned: {total} requests.")
    if total > QUOTA:
        print(
            f"WARNING: Planned requests ({total}) exceed quota ({QUOTA}). Script will stop at {QUOTA}."
        )

    for site_name, (lat, lon) in SITES.items():
        print(f"\n=== {site_name.upper()} ({lat}, {lon}) ===")
        for year in YEARS:
            for month in range(1, 13):
                done += 1
                print(f"[{done}/{total}]", end=" ")
                ok = download_month(site_name, lat, lon, year, month)
                if not ok and requests_made >= QUOTA:
                    break
            if requests_made >= QUOTA:
                break
        if requests_made >= QUOTA:
            break

    print(
        f"\nDone. {requests_made} requests used. Files saved to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
