"""
Explore MetMalaysia API - find location IDs for KL, Penang, KK and available datatypes.
Token: set METMALAYSIA_TOKEN in .env
Usage: python explore_metmalaysia.py
"""

import os
import requests
from dotenv import load_dotenv
from datetime import date

load_dotenv()

TOKEN = os.getenv("METMALAYSIA_TOKEN")
BASE = "https://api.met.gov.my/v2.1"
HEADERS = {"Authorization": f"METToken {TOKEN}"}

# Exact substrings to match (no short abbreviations that cause false positives)
TARGETS = ["kuala lumpur", "pulau pinang", "kota kinabalu", "george town", "georgetown"]


def get_all(endpoint, params):
    """Fetch all records handling 50-record pagination."""
    results = []
    offset = 0
    while True:
        r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params={**params, "offset": offset})
        r.raise_for_status()
        data = r.json()
        batch = data.get("results", [])
        results.extend(batch)
        count = data["metadata"]["resultset"]["count"]
        offset += len(batch)
        if offset >= count or not batch:
            break
    return results


def find_my_locations():
    print("=== All TOWN locations (searching for KL, Penang, KK) ===")
    towns = get_all("locations", {"locationcategoryid": "TOWN"})
    print(f"  Total towns fetched: {len(towns)}")
    matched = [t for t in towns if any(k in t.get("name", "").lower() for k in TARGETS)]
    print(f"  Matched:")
    for t in matched:
        print(f"    {t['id']:30s}  {t['name']}")

    if not matched:
        print("\n  No exact matches. Dumping all town names for manual inspection:")
        for t in towns:
            print(f"    {t['id']:30s}  {t['name']}")

    return matched


def test_observation(location_id, location_name):
    """Try fetching hourly observations — check if historical data is accessible."""
    today = date.today().isoformat()
    print(f"\n=== Hourly observation test: {location_name} ({location_id}), today={today} ===")
    r = requests.get(f"{BASE}/data", headers=HEADERS, params={
        "datasetid": "OBSERVATION",
        "datacategoryid": "HOURLY",
        "locationid": location_id,
        "start_date": today,
        "end_date": today,
    })
    print(f"  Status: {r.status_code}")
    data = r.json()
    results = data.get("results", [])
    if results:
        for row in results[:10]:
            print(f"  {row.get('date','')[:16]}  {row.get('datatype',''):12s}  {row.get('value','')}  {row.get('attributes',{}).get('unit','')}")
    else:
        print(f"  Response: {data}")


def sample_forecast(location_id, location_name):
    today = date.today().isoformat()
    print(f"\n=== General forecast: {location_name} ({location_id}) ===")
    r = requests.get(f"{BASE}/data", headers=HEADERS, params={
        "datasetid": "FORECAST",
        "datacategoryid": "GENERAL",
        "locationid": location_id,
        "start_date": today,
        "end_date": today,
    })
    r.raise_for_status()
    for row in r.json().get("results", []):
        print(f"  {row['datatype']:10s}  {row['value']}  {row['attributes'].get('unit','')}")


if __name__ == "__main__":
    matched = find_my_locations()

    if matched:
        kl = next((t for t in matched if "lumpur" in t["name"].lower()), matched[0])
        sample_forecast(kl["id"], kl["name"])
        test_observation(kl["id"], kl["name"])
    else:
        print("\nNo matched locations — check the full dump above and update TARGETS.")
