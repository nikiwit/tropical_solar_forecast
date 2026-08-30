# Tropical Solar Forecast — Working Documentation

Last updated: 2026-08-30

---

## Evaluation Framework (`src/solarfc/`)

Built and frozen **before** any model is trained, so every baseline, every ablation
variant and the C++ engine are scored by identical code.

```bash
/opt/anaconda3/envs/fyp/bin/python -m pytest      # 73 tests
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
FS = 0.08 at 20min rising to 0.51 at 6h — then collapses to 0.003 at 24h, because naive
persistence at 24h is the same time of day yesterday.

**Ramp metrics exist because RMSE hides the events that matter.** Ramps are ~15% of daytime
samples at KL, so RMSE is dominated by unremarkable timesteps and a model can post an
excellent RMSE while missing every convective collapse. This is also the test Module A must
pass — if the monsoon gate works anywhere it should show up here.

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
| Repo visibility | Public. **No thesis prose in the repo** — Turnitin would match the thesis against its own public copy |

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
- [ ] ERA5 upsampling pipeline (hourly → 10-min; `xarray` + `netCDF4` now installed)
- [ ] Known-future covariate builder — NWP-free and perfect-foresight tracks
- [ ] Feature engineering pipeline (lags, rolling stats, calendar encodings)
- [ ] XGBoost / LightGBM (direct, 9 models per horizon)
- [ ] LSTM encoder-decoder, BiLSTM-GRU, physics-guided CNN-BiLSTM (MIMO)
- [ ] Chronos-2 zero-shot baseline
- [ ] Dataset redistribution licence check (determines Phase 8 release form)

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
