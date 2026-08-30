"""
Download ERA5 hourly single-level data (2016-2020) for Southeast Asia.
Downloads month-by-month to stay within CDS size limits.

Prerequisites:
    1. Register at https://cds.climate.copernicus.eu/
    2. Accept the ERA5 licence at:
       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    3. Get your personal access token from:
       https://cds.climate.copernicus.eu/profile
    4. Create ~/.cdsapirc with:
          url: https://cds.climate.copernicus.eu/api
          key: YOUR-PERSONAL-ACCESS-TOKEN
    5. pip install "cdsapi>=0.7.7"

Usage:
    python download_era5.py
"""

import os
import cdsapi

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "era5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bounding box covering all 7 sites (North/West/South/East)
AREA = [16, 98, -8, 122]  # N, W, S, E

VARIABLES = [
    "total_cloud_cover",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_column_water_vapour",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]

YEARS = [2016, 2017, 2018, 2019, 2020]
MONTHS = list(range(1, 13))
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]

client = cdsapi.Client()


def download_month(year, month):
    output_file = os.path.join(OUTPUT_DIR, f"era5_sea_{year}_{month:02d}.nc")
    if os.path.exists(output_file):
        print(f"  [SKIP] {output_file} already exists")
        return

    print(f"  [REQUESTING] {year}-{month:02d} ...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": DAYS,
            "time": HOURS,
            "area": AREA,
            "data_format": "netcdf",
        },
        output_file,
    )
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  [OK] {output_file} ({size_mb:.1f} MB)")


def main():
    total = len(YEARS) * len(MONTHS)
    done = 0
    for year in YEARS:
        print(f"\n=== {year} ===")
        for month in MONTHS:
            done += 1
            print(f"[{done}/{total}]", end="")
            download_month(year, month)

    print(f"\nDone. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
