# Tropical Solar Forecast — Working Documentation

Last updated: 2026-08-30 (Phase 2 infrastructure complete)

> **The planning document is not in this repository.** It lives at
> `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Second Brain/6 - University/FYP/FYP Plan.md`
> and holds the contributions, novelty positioning, literature, phase actions and
> risk register.
>
> Anything needed to *build* the project is in this file. Anything needed to
> *write about* it is in the plan.

---

## Deployment Requirements (why the design is shaped this way)

Malaysian Large Scale Solar operators are legally required to forecast, under the
Energy Commission's [Guidelines on Large Scale Solar PV Plant](https://www.st.gov.my/sites/default/files/2026-02/Guidelines-on-Large-Scale-Solar-PV-Plant-for-Connection-to-Electricity-Network.pdf),
Appendix B §10. These are obligations on the operator, not preferences.

| Submission | Unit | Interval | Cadence |
|---|---|---|---|
| Rolling 24 Hours Forecast | **MWac** | **15 min** | 24 h ahead, updated **every half hour**, via web service to NLDC |
| Declared Daily Capacity | MWac | 15 min | **Day-ahead by 10:00** — so 38 h out |
| 9-day ahead | MWac | 15 min | Wednesdays before 12:30 |

Site instrumentation: 1 pyranometer + 1 full weather station per 10 MWac
(per 1 MW if distribution-connected), IEC 61724 Class A, ≥15-minute logging,
telemetered via IEC 60870-5-104.

**What this forced in the design:**

| Requirement | Consequence |
|---|---|
| Day-ahead by 10:00 must reach 38 h | Horizons extended to **36 h and 48 h**. The old 24 h ceiling could not produce the required submission |
| 15-minute mandated interval | Train at 10-min (preserves ramp structure), **report at both** |
| Forecast must be MWac | **Contribution 4** — pvlib GHI→MWac chain as a deployment layer |
| Web service is mandatory | REST endpoint is a **core deliverable**, no longer deferrable |
| Plants have no AOD/ozone/cloud-type | **DEPLOYABLE vs FULL feature sets**, both reported |

> **The feature gap is the one that would silently kill a deployment.** NSRDB
> supplies aerosol optical depth, ozone, asymmetry and a cloud-type
> classification. No plant can measure any of these. A model that leans on them
> scores well here and cannot run on site, and no accuracy table would reveal it.
> See `config.SATELLITE_ONLY_FEATURES` and `config.DEPLOYABLE_NSRDB_FEATURES`.

**Cross-checked against AEMO (Australia)** to confirm these are representative
rather than a local quirk. They are, and AEMO is stricter (5-minute dispatch).
AEMO's accreditation test is **MAE_self ≤ MAE_incumbent *and* RMSE_self ≤
RMSE_incumbent** — both required. So report MAE and RMSE jointly per horizon and
never trade one against the other. AEMO also excludes night intervals from
assessment, independently confirming the daytime-masking decision in `metrics.py`.

---

## Evaluation Framework (`src/solarfc/`)

Built and frozen **before** any model is trained, so every baseline, every ablation
variant and the C++ engine are scored by identical code.

```bash
/opt/anaconda3/envs/fyp/bin/python -m pytest      # 237 tests
```

> **Note:** `conda activate fyp` can silently fall through to `virt_env` depending on
> shell profile state. Use the absolute interpreter path `/opt/anaconda3/envs/fyp/bin/python`
> or verify with `which python` after activating.

| Module | Contents |
|---|---|
| `config.py` | Sites, horizons, monsoon phases, split years, thresholds. Single source of truth |
| `data.py` | NSRDB loading, UTC index assembly, grid-continuity checks |
| `splits.py` | Chronological splits, monsoon labelling, transition windows, hashed split manifest |
| `metrics.py` | MAE, RMSE, MBE, MAPE, nMAE, nRMSE, R², Forecast Skill, pinball, PICP, PINAW, reliability |
| `ramp.py` | Ramp-event detection, precision/recall/F1, detection lead time |
| `baselines.py` | Naive persistence, smart persistence, clear-sky, clear-sky index |
| `era5.py` | ERA5 zip handling, nearest-gridpoint extraction, upsampling, unit derivation |
| `covariates.py` | Known-future decoder inputs in three NWP tracks, forecast-error degradation |
| `clearsky.py` | Per-site Linke turbidity fit, calibrated Ineichen envelope |
| `features.py` | Observed-past features — lags, rolling stats, CSI, `delta_csi`, FULL/DEPLOYABLE |
| `scaling.py` | Per-site z-score, fitted on train years, serialised for SolarInfer |
| `dataset.py` | Supervised assembly — joins encoder and decoder sides per horizon |
| `results.py` | The one results schema every phase appends to |
| `models/gbdt.py` | XGBoost and LightGBM, direct multi-horizon |

### Design decisions worth knowing

**Daytime masking is explicit, never implicit.** Night-time GHI is identically zero and
trivially predictable — including it shrinks MAE, inflates R² toward 1, and makes MAPE
undefined. `metrics.daytime_mask()` filters on *clear-sky* GHI > 20 W/m², so the mask is a
property of solar geometry alone and is therefore identical for every model compared.
Metrics never filter silently: a metric that changes its own sample set is not comparable.

**Smart persistence is the primary Forecast Skill reference.** Naive persistence
(`forecast(t+h) = obs(t)`) is weak in a strongly diurnal signal and inflates FS for every
model. Smart persistence holds the clear-sky index constant and rescales by clear-sky
irradiance at the target time. Measured on KL 2020, smart persistence beats naive by
FS = 0.075 at 20min rising to 0.472 at 6h — then collapses to 0.003 at 24h, because naive
persistence at 24h is the same time of day yesterday.

**Ramp metrics exist because RMSE hides the events that matter.** Ramps are ~15% of daytime
samples at KL, so RMSE is dominated by unremarkable timesteps and a model can post an
excellent RMSE while missing every convective collapse. This is also the test Module A must
pass — if the monsoon gate works anywhere it should show up here.

---

## Clear-Sky Calibration (and what it exposed)

`scripts/fit_turbidity.py` — run once, output committed.

pvlib's default Linke turbidity climatology **runs 4.4–7.6% low** at all seven
sites. Since that envelope is the denominator of every clear-sky index here, the
default put a systematic multiplicative error straight into the target.

| Site | pvlib default | Fitted | RMSE on clear periods |
|---|---|---|---|
| kuala_lumpur | 4.69 | **3.917** | 17.3 W/m² |
| penang | 4.19 | **3.969** | 17.3 |
| kota_kinabalu | 4.39 | **3.736** | 15.0 |
| ho_chi_minh | 4.74 | **3.788** | 22.2 |
| bangkok | 4.51 | **4.258** | 26.2 |
| jakarta | 4.46 | **4.269** | 20.1 |
| manila | 4.09 | **3.417** | 17.0 |

Fitted against observed GHI on Reno-Hansen detected-clear periods — what a plant
could do from its own pyranometer, so the calibration stays inside the
DEPLOYABLE story. Cross-checked against a grid search on NSRDB clear-sky: the
two agree within 0.18 at six of seven sites.

> **Ineichen, not McClear.** Yang (2020) compares Ineichen-Perez, McClear and
> REST2 and finds forecast RMSE is essentially unchanged by the choice, so it
> should be made on accessibility. He recommends McClear — but that is a web
> service, and SolarInfer runs offline on a Pi. Ineichen plus one fitted float
> per site is the accessible option *here*.

### Finding: NSRDB cannot represent cloud enhancement

Max GHI / clear-sky ratio is **exactly 1.000000** at every site, with **20–34%
of daytime samples identically equal**. Solcast behaves the same way (max
exactly 1.000). Both are transmittance retrievals, `GHI = clearsky × τ`, `τ ≤ 1`.

Two consequences for Chapter 4:

1. The clear-sky index here **is** retrieved cloud transmittance. Values above 1
   against the fitted envelope are calibration residue, not over-irradiance.
2. **The benchmark cannot evaluate cloud-enhancement events**, which produce
   some of the sharpest positive ramps in tropical conditions. Ramp results
   cover downward and moderate upward ramps only. This is a property of the
   evaluation target, not of any model tested.

The CSI clip was therefore raised 1.5 → **2.0**: it was written to preserve
physics that provably does not occur in this data. At 2.0 it binds on
0.09–0.29% per site.

Also noted: the existing daytime mask (`clear-sky > 20 W/m²`) is **equivalent to
the field's standard zenith < 85° cutoff** — at KL they select 123,569 and
123,988 samples. Same filter, different units. No change needed.

### Specification bugs found during implementation

| Issue | Resolution |
|---|---|
| **15-min horizon not representable** on a 10-min grid (1.5 steps) | Replaced with **20 min** (2 steps). Keeps nine horizons and stays above the 15-min nowcasting boundary, so the plan's scope statement is unaffected. Horizons are now defined in *steps* in `config.HORIZON_STEPS` |
| **Smart persistence undefined at 12h** — every daytime target has a night origin, where CSI is a ratio against zero. The whole horizon evaluated to NaN, silently deleting the FS reference | Persist the last *observed* CSI across the night (`carry_overnight=True`, default). Strict instantaneous form retained as an option |
| **Transition windows covered 40% of the year** at ±21 days — April and May windows merged, destroying the contrast | Narrowed to **±10 days** → disjoint windows over ~23% of the year |

### Verified against real data (KL 2020)

- 52,704 rows, zero missing, uniform 10-minute grid, no irregular steps
- 25,639 daytime samples (48.6%)
- Ramp base rate 0.147 — by phase: NE 0.153, Inter-I 0.140, SW 0.147, Inter-II 0.124

---

## Known-Future Covariates and NWP Error

### Three tracks, not two

Every model trains and reports on three known-future regimes. The decoder spans
the full 48 h maximum horizon in one pass with all horizons read from it, which is
standard TFT and the natural design for a MIMO head.

| Track | Known-future atmospheric inputs | What it measures |
|---|---|---|
| `nwp_free` | none — calendar and solar geometry only | Honest lower bound |
| `realistic` | ERA5 degraded by **measured** forecast error | **The operational number** |
| `perfect` | ERA5 unmodified | Optimistic ceiling |

Reporting perfect-foresight numbers as operational accuracy is the most common
criticism of solar forecasting papers, so **every results table must state which
track it reports.**

### The degradation is measured, not assumed

Published NWP verification is overwhelmingly European or CONUS, where convective
cloud behaves differently. Rather than borrow those numbers, forecast error was
measured from archived JMA GSM forecasts over all seven sites, 2018–2020, via
[Open-Meteo's Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api).

```bash
python scripts/pull_jma_forecasts.py           # ~4 min, 21 requests
python -c "from solarfc.nwp_error import fit_all_sites; ..."
```

Fitted models live in `data/processed/nwp_error/error_models.json` (committed —
they are results, not raw data) and load via
`covariates.load_fitted_error_models()`.

**Two error definitions, both reported:**

- **drift** — JMA at lead L vs its own lead-0 run. Model bias and resolution
  cancel, isolating pure forecast decay. The clean number to quote for NWP skill.
- **total** — JMA vs ERA5. What a deployed model actually experiences: trained on
  ERA5, fed a real forecast, so the input differs by decay *plus* bias *plus*
  resolution. Measured at **1.2–2.3× drift**, and it is what calibrates the
  realistic track. Drift would leave the "realistic" track closer to
  perfect-foresight than to reality.

**Measured error (std, pooled across 7 sites, total mode):**

| Variable | 24 h | 48 h | 72 h |
|---|---|---|---|
| Cloud cover (fraction) | 0.277 | 0.290 | 0.301 |
| Temperature (K) | 1.33 | 1.36 | 1.39 |
| Dewpoint (K) | 1.14 | 1.25 | 1.36 |
| Relative humidity (%) | 8.0 | 8.4 | 8.7 |
| Precipitation (mm/h) | 0.71 | 0.73 | 0.74 |

### Finding: forecast skill has a latitude gradient

JMA cloud-forecast correlation against ERA5, and how it decays:

| Site | lat | r @0h | r @24h | r @72h |
|---|---|---|---|---|
| Kuala Lumpur | 3.1 | 0.512 | 0.456 | 0.365 |
| Penang | 5.4 | 0.532 | 0.480 | 0.417 |
| Kota Kinabalu | 6.0 | 0.548 | 0.492 | 0.420 |
| Jakarta | −6.2 | 0.579 | 0.505 | 0.429 |
| Ho Chi Minh | 10.8 | 0.507 | 0.459 | 0.393 |
| Bangkok | 13.8 | 0.649 | 0.643 | 0.599 |
| Manila | 14.6 | 0.720 | 0.671 | 0.614 |

**Correlation between |latitude| and 24 h skill: 0.834.** Equatorial sites
(|lat| < 7) average r = 0.483 at 24 h; off-equator sites average 0.591.

Physically coherent: equatorial convection is locally driven and chaotic at
mesoscale, while synoptic monsoon dynamics further from the equator are more
predictable. This supports the project's premise empirically rather than by
assertion — and it is a reportable result, since NWP verification for equatorial
SEA is thinly published.

> **Do not overstate this.** JMA's own *analysis* correlates only 0.578 with ERA5
> cloud, so a large share of the disagreement is inter-model representation, not
> forecast failure. A skill-score framing initially suggested "near-zero skill",
> which was an artefact of that. Forecast decay itself is modest: 0.578 → 0.529 at
> 24 h → 0.463 at 72 h.

### Known limitation: no operational NWP GHI baseline on the test year

JMA GSM carries **no shortwave radiation at any date**, and archived forecast GHI
begins in **2024** for every model (ECMWF, GFS, ICON). NSRDB ends in **2020**.
There is no overlap and there never will be.

Consequence: the operational NWP baseline splits in two —

1. **2020 benchmark:** a cloud-driven GHI baseline built from JMA's archived
   forecast cloud via a clear-sky transmittance model. The cloud input is a
   genuine forecast, so label it a *cloud-driven NWP baseline*, not an
   operational GHI forecast.
2. **Phase 6/7 demo:** real archived ECMWF/GFS GHI forecasts for 2024+ against
   Solcast actuals. This is the true incumbent comparison.

---

## Project Setup

### Repository
- **GitHub:** github.com/nikiwit/tropical_solar_forecast (public)
- **Local:** `/Users/kita/Library/Mobile Documents/com~apple~CloudDocs/APU/FYP/Code/`

### Conda Environment
```bash
conda activate fyp          # Python 3.13
```

Installed packages:
- `torch` 2.11.0, `pandas` 3.0.2, `numpy`, `matplotlib`, `seaborn`
- `pvlib`, `optuna`, `shap`, `xgboost`, `lightgbm`
- `solcast`, `requests`, `python-dotenv`, `cdsapi`
- `jupyter`

### Installation

```bash
/opt/anaconda3/envs/fyp/bin/python -m pip install -e .          # framework + data pipeline
/opt/anaconda3/envs/fyp/bin/python -m pip install -e ".[models]" # + torch, xgboost, lightgbm, optuna, shap
```

Editable src-layout install, so `import solarfc` works from anywhere — scripts,
notebooks, tests — with no `sys.path` manipulation.

### Directory Structure
```
Code/
├── .env                    # API keys (gitignored)
├── .gitignore
├── pyproject.toml          # package metadata, deps, pytest config
├── DOCS.md                 # this file
├── src/solarfc/            # the package
│   ├── config.py           # sites, horizons, monsoon phases, splits, thresholds
│   ├── data.py             # NSRDB loading, continuity checks
│   ├── splits.py           # chronological splits, monsoon labels, transitions
│   ├── metrics.py          # point, skill and probabilistic metrics
│   ├── ramp.py             # ramp-event detection and scoring
│   └── baselines.py        # persistence and clear-sky references
├── scripts/                # data acquisition (run once, not imported)
│   ├── download_nsrdb.py
│   ├── download_era5.py
│   ├── download_solcast.py
│   └── explore_metmalaysia.py
├── data/                   # all downloaded data (gitignored)
│   ├── nsrdb/              # 7 sites × 5 years = 35 CSVs
│   ├── era5/               # 12 months × 5 years = 60 NetCDFs
│   └── solcast/            # KL + Penang 2019 = 24 CSVs
├── notebooks/              # EDA, visualisation
├── configs/                # experiment / hyperparameter YAMLs
├── results/                # metrics tables, figures (gitignored output)
├── tests/                  # pytest suite
└── solarinfer/             # C++ inference engine (Phase 6)
    ├── src/
    ├── include/
    └── tests/
```

### Conventions

| Area | Convention |
|---|---|
| Branching | Work on `development`. Merge to `main` at phase completion and tag (`phase-2-baselines`). Feature branches only for large or risky work |
| Commits | [Conventional Commits](https://www.conventionalcommits.org/) — `type(scope): subject`, with a body explaining *why*. Scope is the module or phase |
| Docstrings | NumPy style, matching numpy/scipy/pandas/pvlib. Rationale paragraphs are kept deliberately — they become Chapter 4 methodology text |
| Imports | Always `from solarfc import ...`. Never `sys.path` hacks |
| Scope | Code, data pipelines and build documentation only. Thesis writing lives in the plan document |

---

## API Credentials

| Service | Location | Format |
|---|---|---|
| NREL (NSRDB) | `.env` | `NREL_API_KEY=...` and `NREL_EMAIL=...` |
| Copernicus CDS (ERA5) | `~/.cdsapirc` | `url:` and `key:` lines |
| MetMalaysia | `.env` | `METMALAYSIA_TOKEN=...` |

All are gitignored / outside repo.

---

## Data Downloads

### NSRDB Himawari PSM v3

**What:** Satellite-derived solar irradiance + weather at 10-minute resolution, 2016–2020.

**Script:** `download_nsrdb.py`

**Run:**
```bash
conda activate fyp
python download_nsrdb.py
```

**Output:** `data/nsrdb/{site}_{year}.csv` (35 files)

**Sites:**
| Site | Lat | Lon |
|---|---|---|
| kuala_lumpur | 3.139 | 101.687 |
| penang | 5.414 | 100.330 |
| kota_kinabalu | 5.980 | 116.073 |
| ho_chi_minh | 10.823 | 106.630 |
| bangkok | 13.754 | 100.501 |
| jakarta | -6.208 | 106.846 |
| manila | 14.599 | 120.984 |

**Variables downloaded:** ghi, dni, dhi, clearsky_ghi/dni/dhi, air_temperature, dew_point, relative_humidity, surface_pressure, wind_speed, wind_direction, total_precipitable_water, surface_albedo, cloud_type, fill_flag, solar_zenith_angle, aod, alpha, ozone, asymmetry

**Rate limits:** max 1 request per 2 seconds (CSV). Script handles this automatically.

**Note:** `cld_opd_dcomp` (cloud optical depth) is NOT available in the Himawari API — don't try to request it.

### ERA5 Reanalysis

**What:** Hourly NWP features at ~31 km resolution covering all of SEA, 2016–2020. Will be upsampled to 10-min in preprocessing.

**Script:** `download_era5.py`

**Prerequisites:**
1. Register at https://cds.climate.copernicus.eu/
2. Accept ERA5 licence on dataset page
3. Get token from profile page
4. Token goes in `~/.cdsapirc` (NOT in .env)

**Run:**
```bash
conda activate fyp
python download_era5.py
```

**Output:** `data/era5/era5_sea_{year}_{month}.nc` (60 NetCDF files — monthly granularity, 12 months × 5 years)

**Bounding box:** 16°N to 8°S, 98°E to 122°E (covers all 7 sites)

**Variables:** total_cloud_cover, 2m_temperature, 2m_dewpoint_temperature, mean_sea_level_pressure, 10m wind (u/v), total_column_water_vapour, total_precipitation, surface_solar_radiation_downwards

**Note:** ERA5 requests are queued server-side. Each year can take minutes to hours. Script waits automatically.

### Solcast Historical Radiation

**What:** High-accuracy satellite-derived solar irradiance at 30-minute resolution. Used for spot-validation against NSRDB data, not for model training.

**Script:** `download_solcast.py`

**Prerequisites:**

1. Register at [solcast.com/data-for-researchers](https://solcast.com/data-for-researchers) (use university email)
2. Get API key from account settings
3. Add `SOLCAST_API_KEY=your_key` to `.env`
4. `pip install solcast pandas`

**Run:**
```bash
conda activate fyp
python download_solcast.py
```

**Output:** `data/solcast/{site}_{year}_{month}.csv` (24 files — KL + Penang, 2019, monthly)

**Quota:** 50 free requests total (researcher account). 24 used, 26 remaining.

**Variables:** ghi, clearsky_ghi, dni, dhi, cloud_opacity, air_temp, relative_humidity, wind_speed_10m, wind_direction_10m, precipitable_water

**Note:** To extend coverage, edit `YEARS` or `SITES` in the script. The hard quota guard will stop the script before exceeding 50 requests.

---

## Progress

### Phase 1: Setup & Data Acquisition — Near Complete (as of 2026-04-06)

- [x] NREL API key registered
- [x] Copernicus CDS account registered
- [x] Solcast researcher account registered
- [x] Conda environment `fyp` created (Python 3.11)
- [x] NSRDB download script written and tested
- [x] ERA5 download script written
- [x] Solcast download script written (`download_solcast.py`)
- [x] GitHub repo created (`main` + `development` branches)
- [x] Project directory structure set up
- [x] NSRDB downloads complete — 35 CSVs (7 sites × 5 years, 2016–2020)
- [x] ERA5 downloads complete — 60 NetCDFs (12 months × 5 years, monthly granularity)
- [x] Solcast downloads complete — 24 CSVs (KL + Penang, 2019, monthly) — 24/50 requests used
- [x] `pandas` (2.2.2), `numpy` (2.2.2), `matplotlib` (3.9.0), `seaborn` (0.13.2) installed
- [x] `torch` (2.7.0) installed — MPS backend available on M2
- [x] `pvlib`, `optuna`, `shap`, `xgboost`, `lightgbm` installed
- [ ] EDA notebooks (irradiance distributions, clear-sky index, monsoon transitions)
- [x] MetMalaysia API access obtained — forecast-only, see MetMalaysia section below
- [ ] University HPC access — using M2 32GB + Google Colab Pro as needed

---

## MetMalaysia API

**Base URL:** `https://api.met.gov.my/v2.1/`  
**Auth:** `Authorization: METToken <token>` — token stored in `.env` as `METMALAYSIA_TOKEN`  
**Rate limits:** 10 requests/min burst, 2,000 requests/day  
**Docs:** [api.met.gov.my](https://api.met.gov.my) (official)

### Access level (default upon registration)

- General weather forecast
- Marine forecast
- Warnings

Hourly observations (TEMP, RH, WND_SPD, etc.) return **403** — not included in the default plan.

### Endpoints

| Endpoint | Description |
| --- | --- |
| `/locations?locationcategoryid=TOWN` | List towns (paginated, 50/page, use `&offset=N`) |
| `/datatypes` | List all data type IDs |
| `/data` | Fetch forecast or observation data |

### Location IDs for FYP sites

| Site | Location ID | Notes |
| --- | --- | --- |
| Kuala Lumpur | `LOCATION:340` | TOWN category |
| George Town (Penang) | `LOCATION:210` | TOWN category — listed as "GEORGETOWN" |
| Kota Kinabalu | — | Not found in TOWN list (154 towns exhausted) |

### Forecast data types available

| ID | Description |
| --- | --- |
| `FGM` / `FGA` / `FGN` | General forecast text (morning / afternoon / night) |
| `FMAXT` / `FMINT` | Daily max / min temperature (°C) |
| `FSIGW` | Significant weather description |

### Example request (KL general forecast)

```text
GET /v2.1/data?datasetid=FORECAST&datacategoryid=GENERAL&locationid=LOCATION:340&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
```

`start_date` must be today or later — **no historical data**.

### Relevance to FYP

Not useful for training (no historical obs, no solar radiation data). Intended use: **SolarInfer live demo** — weather summary panel for KL and Penang alongside model GHI output. Deferred to Phase 6.

**Exploration script:** `explore_metmalaysia.py`

---

### Phase 2: Baselines — In Progress (started 2026-08-30)

- [x] Evaluation framework (`src/solarfc/`) — metrics, splits, ramp scoring, baselines
- [x] 73 unit tests passing, including leakage guards on every baseline
- [x] Naive persistence, smart persistence, clear-sky baselines implemented and verified on real data
- [x] Hashed split manifest writer (feeds the Zenodo deposit)
- [x] ERA5 upsampling pipeline — zip handling, nearest gridpoint, accumulation shift verified
- [x] ERA5 cache built to Parquet, 7 sites × 263,083 rows, validated against NSRDB
- [x] Known-future covariate builder — three tracks (NWP-free, realistic, perfect-foresight)
- [x] Measure real NWP error from JMA archived forecasts, fit the degradation model
- [x] Per-site Linke turbidity calibration (`clearsky.py`, artefact committed)
- [x] Feature engineering pipeline (lags, rolling stats, FULL vs DEPLOYABLE sets)
- [x] Per-site z-score standardiser, serialisable for SolarInfer
- [x] Supervised assembly with leakage guards asserted in tests
- [x] Results schema — one long-format table for every phase
- [x] Reproducible baselines script (replaces the interactive reference table)
- [x] XGBoost / LightGBM trainers + resumable run harness
- [ ] **Run the full XGBoost grid** (924 models — both targets, then freeze)
- [ ] LightGBM on the winning target (462 models)
- [ ] Operational NWP baseline (JMA GSM archived GHI)
- [ ] LSTM encoder-decoder, BiLSTM-GRU, physics-guided CNN-BiLSTM (MIMO)
- [ ] Chronos-2 zero-shot baseline
- [ ] Cloud-driven NWP baseline for the 2020 benchmark
- [ ] Dataset redistribution licence check (determines Phase 8 release form)

---

## Resume Here

**State as of 2026-08-30:** Phase 1 complete, Phase 2 infrastructure complete
and validated. The full training grid has **not** been run yet.

### Run this first

```bash
/opt/anaconda3/envs/fyp/bin/python -m pytest      # expect 237 passing
```

### The Phase 2 run order

```bash
# 1. Clear-sky calibration — once, ~20s. Output is committed.
python scripts/fit_turbidity.py

# 2. Reference baselines — ~2-3 min for all 7 sites.
python scripts/run_baselines.py

# 3. Smoke test the trainer before committing to the full grid — ~30s.
python scripts/train_gbdt.py --smoke

# 4. Full XGBoost grid: 7 sites x 11 horizons x 3 tracks x 2 feature
#    sets x 2 targets = 924 models, roughly 1-1.5 h. Resumable.
python scripts/train_gbdt.py

# 5. LightGBM on whichever target won, once step 4 reports it.
python scripts/train_gbdt.py --algorithm lightgbm --targets csi
```

Everything appends to `data/processed/results/results.csv`. The harness skips
combinations already present, so an interrupted run continues where it stopped
rather than starting over.

`conda activate fyp` can silently fall through to `virt_env` depending on shell
state. Use the absolute interpreter path, or check `which python` after
activating.

### What exists and is validated

| Component | Evidence |
|---|---|
| Evaluation framework | 237 tests, metrics frozen before any model |
| ERA5 pipeline + cache | 7 sites × 263,083 rows; ssrd vs NSRDB GHI r = 0.897–0.946 |
| Known-future covariates | Three tracks, leakage guards asserted in tests |
| NWP error models | Fitted to measured JMA error, validated to within 0.2% |
| Clear-sky calibration | 7 sites fitted; two independent methods agree within 0.18 at 6/7 |
| Feature pipeline | 68 FULL / 58 DEPLOYABLE columns; tamper test proves no future leaks back |
| GBDT harness | Tracks order correctly at every horizon; beats climatology at 6h and 36h |

**Track separation on KL (the check that the whole design rests on)** — MAE,
`csi` target, FULL features:

| Horizon | nwp_free | realistic | perfect |
|---|---|---|---|
| 20 min | 64.5 | 64.6 | 64.2 |
| 6 h | 109.2 | 105.6 | 101.4 |
| 36 h | 116.1 | 108.0 | 102.3 |

A weather forecast buys nothing 20 minutes out and a great deal at 36 h, with
`realistic` sitting properly between the bounds — which is what the fitted
degradation model was built to produce.

Cached data is gitignored and must be rebuilt on a fresh machine:

```bash
python scripts/build_era5_cache.py     # ~4.5 min, needs data/era5/*.nc
python scripts/pull_jma_forecasts.py   # ~4.5 min, network only
```

The fitted error models are committed, so the realistic track works without
re-running the JMA pull.

### Next task

**Feature pipeline, then XGBoost.** That produces the first real accuracy number
and tells you whether the data supports the horizon set at all — before any
Transformer work. If XGBoost cannot beat smart persistence at 6 h, nothing
downstream will either.

Reference numbers to beat — KL 2020, daytime only, regenerate with
`python scripts/run_baselines.py`. **These supersede the numbers previously
recorded here**, which were produced interactively and used NSRDB's clear-sky
envelope for the daytime mask; the mask is now the fitted Ineichen envelope, so
the sample set differs slightly.

Two smart-persistence variants. The NSRDB one is the **primary FS reference** —
FS is a ratio, so the honest choice is the strongest available reference. The
Ineichen one is what a plant could run without a satellite, and the gap between
them is the clear-sky calibration penalty.

| Horizon | MAE | RMSE | FS vs naive | MAE (Ineichen) | Gap |
|---|---|---|---|---|---|
| 20 min | 63.7 | 112.5 | 0.075 | 65.6 | +1.8 |
| 30 min | 75.1 | 126.5 | 0.121 | 77.3 | +2.2 |
| 1 h | 96.8 | 151.9 | 0.244 | 99.8 | +3.0 |
| 2 h | 125.6 | 186.5 | 0.388 | 130.6 | +5.0 |
| 3 h | 148.8 | 215.7 | 0.450 | 155.6 | +6.8 |
| 6 h | 196.2 | 276.8 | 0.472 | 209.0 | +12.8 |
| 12 h | 207.4 | 287.5 | 0.440 | 225.0 | +17.7 |
| 18 h | 196.1 | 277.7 | 0.435 | 210.9 | +14.8 |
| 24 h | 148.0 | 209.8 | 0.003 | 149.2 | +1.2 |
| 36 h | 209.1 | 290.8 | 0.434 | 226.6 | +17.4 |
| 48 h | 149.2 | 210.0 | 0.004 | 150.7 | +1.6 |

FS vs naive collapses at 24 h and 48 h because naive persistence at exactly 24 h
is the same clock time yesterday, which is a strong reference. The gap column
peaks at the horizons whose forecast origin falls at night (12 h, 36 h), where
the clear-sky envelope has to be extrapolated furthest.

### Open items needing a decision or an action

- **SEDA PVMS request** — drafted at `../SEDA-PVMS-data-request-DRAFT.md`, held
  until a supervisor is assigned. Blocks nothing; send by Phase 4
- **Nothing pushed to origin yet** — 11 commits sitting on `development`
- **To verify:** NSRDB Himawari coverage at Darwin; Kaggle India plant scale;
  DKASC co-located irradiance

### Phase 3: Standard TFT + Transformer Baselines — Not Started

Includes: Standard TFT, PatchTST (Nie et al. 2023), iTransformer (Liu et al. 2024). Both PatchTST and iTransformer evaluated at L=72 (aligned) and L=288 (their optimum). Lookback window: ≥72 steps (12h at 10-min).

### Phase 4: Tropical-TFT — Not Started

### Phase 5: Evaluation & Benchmarking — Not Started

### Phase 6: SolarInfer (C++) — Not Started

---

## Validation Strategy

Three distinct validation layers, serving different purposes:

### 1. Academic Benchmark (Chapter 5 — Results)

Train/val/test split strictly by time — no shuffling:

| Split | Years | Purpose |
|---|---|---|
| Training | 2016–2018 | Model learns from |
| Validation | 2019 | Hyperparameter tuning (Optuna) |
| Test | 2020 | Held-out benchmark — reported in thesis |

Metrics per horizon per monsoon split: MAE, RMSE, MAPE, R², Forecast Skill (FS = 1 − RMSE/RMSE_persistence), MBE (per monsoon split — detects seasonal bias), PICP, reliability diagrams.

### 2. C++ Equivalence Check (Chapter 6 — SolarInfer)

Not about weather data — about numerical correctness:
- Feed the same 2020 test inputs to both PyTorch and SolarInfer C++
- Outputs must match to **4 decimal places in normalised (z-score) space**
- This is the SolarInfer validation gate — must pass before engine is considered complete

### 3. Real-World Demo (Chapter 6 — Deployment)

Use remaining Solcast quota (26 requests) to pull post-2020 data (e.g. 2023–2024) for KL:
- Feed into SolarInfer as live input
- Compare forecast output against actual Solcast GHI readings
- Shows the engine working on weather the model was never trained on
- Impressive for thesis presentation and viva

**Plan:** Reserve Solcast quota until after Phase 6. Pull ~12 months of KL data (2024) = 12 requests, leaving 14 in reserve.
