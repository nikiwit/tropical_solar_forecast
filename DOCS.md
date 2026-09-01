# Tropical Solar Forecast — Working Documentation

Last updated: 2026-08-31 (Phase 2 complete)

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
/opt/anaconda3/envs/fyp/bin/python -m pytest      # 265 tests
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
| `nwp_baseline.py` | Cloud-driven and MOS operational NWP incumbent |
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

---

## Phase 2 Results (XGBoost, 924 models, 28m38s, 0 failures)

`data/processed/results/results.csv`. Config: untuned defaults, `lr=0.05`,
`max_depth=8`, early stopping on 2019. Rounds used 36–194 — nowhere near the
2000 cap, so **these are not XGBoost's ceiling**; Optuna tuning is Phase 3.

### Target representation — settled, and frozen in `config.DEFAULT_TARGET`

Clear-sky index won **318 of 462** paired comparisons (68.8%), Wilcoxon
p = 1.4e-20. Margin is small — 0.35 W/m² MAE, ~0.4% — so it is a *reliable*
preference, not a large one.

The win sits exactly where theory says it should. Removing the diurnal cycle
helps most where that cycle dominates:

| Horizon | 20min | 30min | 1h | 2h | 3h | 6h | 24h |
|---|---|---|---|---|---|---|---|
| CSI win rate | **100%** | 93% | 81% | 52% | 43% | 67% | 81% |

Past 2h the model is doing conditional climatology and the representation
stops mattering.

### vs smart persistence — 77 site-horizon pairs

| | Result |
|---|---|
| Beats on RMSE | **77/77** |
| Beats on MAE | 64/77 |
| Beats on **both** (AEMO accreditation test) | **64/77** |

Every one of the 13 failures is at **20 min or 30 min**. From 1 h out, XGBoost
passes both criteria at every site.

| Horizon | XGBoost | Smart persistence | Improvement |
|---|---|---|---|
| 20 min | 58.7 | 55.8 | **−5.1%** |
| 30 min | 67.6 | 66.0 | **−2.3%** |
| 1 h | 82.1 | 85.5 | +4.1% |
| 3 h | 98.2 | 130.3 | +24.7% |
| 6 h | 103.1 | 170.8 | +39.7% |
| 12 h | 106.4 | 184.0 | +42.2% |
| 24 h | 106.4 | 137.5 | +22.7% |
| 36 h | 109.7 | 192.7 | +43.1% |
| 48 h | 109.1 | 144.7 | +24.6% |

> **Why sub-hourly fails MAE but never RMSE.** The models train on squared
> error, so they optimise RMSE at MAE's expense. At 20–30 min persistence is
> near-optimal and that trade is enough to lose the MAE comparison. AEMO
> requires both criteria, so this is not cosmetic — and it is fixable, below.

### Fixing the sub-hourly AEMO failure: use the right loss

Refitting 20 min and 30 min with an absolute-error objective
(`--objective mae`), all 7 sites, `realistic`/FULL/`csi`:

| Objective | MAE pass | RMSE pass | **Both** | Mean MAE |
|---|---|---|---|---|
| `squared_error` (default) | 1/14 | 14/14 | **1/14** | 63.1 |
| `huber` | 2/14 | 14/14 | **2/14** | 62.5 |
| **`mae`** | 12/14 | 14/14 | **12/14** | **59.3** |

Smart persistence averages 60.9 over the same 14 pairs, so the absolute-error
models go from losing to winning. **RMSE compliance is unaffected — 14/14 for
every objective** — so the fix costs nothing on the criterion that already
passed.

Per site at 20 min, `mae` objective vs smart persistence: KL 61.0 v 63.7,
Penang 50.7 v 52.4, KK 48.7 v 49.5, HCMC 56.5 v 57.9, Bangkok 42.1 v 42.4,
Jakarta 56.8 v 58.1 — all wins. **Manila is the sole failure** (69.5 v 66.9),
consistent with it being the most variable site.

Takeaway for the write-up: the loss function is a per-horizon choice, not a
default. Squared error for the horizons where RMSE dominates, absolute error
sub-hourly where persistence is the thing to beat.

### Learning rate: no headroom (negative result)

Rounds stopping at 36–194 suggested `lr=0.05` might be leaving something on the
table. It is not. 7 sites × 4 horizons:

| Learning rate | Mean MAE | Mean RMSE | Typical rounds |
|---|---|---|---|
| 0.05 | 87.56 | 125.49 | ~200 |
| 0.02 | 87.47 | 125.14 | ~300 |
| 0.01 | 87.58 | 125.28 | ~600 |

Differences are ~0.1%, i.e. noise, for 3× the compute. **Optuna should not
spend trials on learning rate.**

### What else was tried, and did not help

| Lever | Outcome |
|---|---|
| Loss function (`mae` sub-hourly) | **The one real gain** — AEMO 1/14 → 12/14 |
| Learning rate 0.05 / 0.02 / 0.01 | Nothing (87.56 / 87.47 / 87.58 MAE) |
| Deeper trees, `max_depth` 8 → 12 | **Worse** at every site and horizon tested |
| `min_child_weight` 5 → 20 | ~1–2% sub-hourly only; 6 h unchanged (122.4→122.1) |
| Dropping Manila's flagged rows | **Worse**, 67.8 → 68.9 — losing 26% of the training data costs more than the retrieval noise it removes |

> **What this adds up to.** Extra satellite features buy nothing past 3 h, error
> is flat from 6 h to 48 h, and no hyperparameter moves the result. Three
> independent findings pointing the same way: the gradient-boosted models sit at
> a **predictability ceiling, not a capacity or feature ceiling**. That is the
> project's founding premise about tropical convection, now measured rather than
> asserted — and it frames what comes next honestly. The encoder side looks
> close to tapped out, so whatever Tropical-TFT gains should come from the
> decoder/NWP path and from regime awareness, which is what Module A is for.

Manila is not fixable by tuning. Best achieved is 67.8 against smart
persistence at 66.9, with the highest sky variability of the seven sites and
26% flagged retrievals. Report it as the hard case rather than chasing it.

### The deployable gap closes past 3 h

DEPLOYABLE − FULL, MAE (W/m², `csi`/`realistic`, mean over 7 sites):

| 20min | 1h | 3h | 6h | 12h | 24h | 36h |
|---|---|---|---|---|---|---|
| +1.27 | +1.09 | +0.85 | +0.35 | −0.21 | −0.17 | −0.15 |

**Satellite-only features (AOD, ozone, asymmetry, cloud type) are worth ~1.3
W/m² at 20 min and nothing at all past 6 h.** Negative values are noise —
DEPLOYABLE is a strict subset, so it cannot genuinely be better.

This is the headline Contribution 4 number: a plant with only IEC 61724
instrumentation loses **nothing measurable** at the horizons the Malaysian grid
code actually mandates.

### Error goes flat after 6 h

KL, `realistic`, FULL: 6h 105.1 → 12h 107.2 → 24h 107.9 → 36h 108.8 → 48h 107.5.
A 48-hour forecast is as accurate as a 6-hour one. Beyond ~6 h the model runs on
conditional climatology plus NWP and extra lead time costs almost nothing.

> **FS swings are the reference moving, not the model.** FS is +0.31 at 24 h and
> +0.50 at 36 h while the model sits at ~108 MAE at both. Smart persistence at
> exactly 24 h is strong (same clock time yesterday, MAE 137) and weak at 36 h
> (night origin, MAE 193). Always quote MAE beside FS.

### Site difficulty at 24 h (`realistic`, FULL, `csi`)

| Site | MAE | nRMSE | R² |
|---|---|---|---|
| bangkok | 86.0 | 27.4% | 0.79 |
| ho_chi_minh | 102.2 | 30.0% | 0.76 |
| kota_kinabalu | 105.9 | 30.7% | 0.77 |
| penang | 105.4 | 32.8% | 0.73 |
| kuala_lumpur | 107.9 | 33.4% | 0.72 |
| jakarta | 111.1 | 35.7% | 0.67 |
| **manila** | **126.2** | **40.6%** | **0.61** |

Manila is the hardest site by a clear margin. Diagnosed rather than assumed —
**site difficulty is predicted almost entirely by sky variability**:

| Property | r with 24 h nRMSE |
|---|---|
| **CSI interquartile range** | **+0.854** |
| CSI standard deviation | +0.709 |
| NSRDB fill-flag rate | +0.692 |
| Overcast fraction (CSI < 0.3) | +0.663 |
| Clear fraction (CSI > 0.95) | −0.551 |
| Ramp rate | +0.333 |

Manila tops the variability measures — widest CSI IQR (0.464 vs 0.327 at
Bangkok), most overcast, highest ramp rate — **and** has the worst retrieval
quality, with 24.2% of daytime samples carrying a non-zero NSRDB fill flag
against 15–20% elsewhere. So part of its error is data quality rather than
forecast difficulty, which matters when interpreting the transfer study.

> **Correction to an earlier note here:** Manila's turbidity fit is *not*
> suspect. Its clear-period RMSE is 17.0 W/m², mid-range for the set, and fit
> quality correlates **negatively** with site error (r = −0.538) — better-fitted
> sites are the harder ones, because a stable sky is both easier to calibrate
> against and easier to forecast. Manila does have the fewest detected clear
> periods (6,100), but that is a symptom of its variability, not a defect in the
> calibration.

### Where this sits against published work

nRMSE 19.7% at 20 min, 29.9% at 6 h, 31.7% at 24 h. Published tropical GHI
forecasting runs roughly 15–25% intra-hour, 20–35% intra-day, 30–45%
day-ahead — so this is inside the normal band throughout and at the good end
day-ahead.

> **Do not compare these directly to ground-station papers.** NSRDB is
> satellite-derived, so it is smoother than a pyranometer and contains no
> enhancement events. Real measured irradiance would score worse. The Phase 5
> Darwin comparison is what quantifies that gap.

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

## LightGBM (444 models, 18m34s, 0 failures)

Run on the frozen `csi` target only. Same grid, same splits, same metrics code.

### It wins, reliably, by almost nothing

Paired against XGBoost over all 462 site × horizon × track × feature-set cells:

| Metric | LightGBM | XGBoost | LightGBM wins | Mean margin | Wilcoxon |
|---|---|---|---|---|---|
| MAE | **94.16** | 94.71 | 395/462 | −0.55 W/m² | p = 1.7e-54 |
| RMSE | **132.60** | 133.13 | 385/462 | −0.53 W/m² | p = 9.6e-51 |

It wins at every single horizon (42/42 cells at 20 min), margin 0.26–0.85 W/m²,
so 0.3–0.9%. Same shape as the target-representation result: a *reliable*
preference, not a meaningful one. Use LightGBM as the tree baseline — it is
also the faster of the two — but do not present the gap as a finding.

AEMO dual-criterion against smart persistence improves 64/77 → **66/77**.
Every remaining failure is still at 20/30 min, so the absolute-error objective
fix carries over unchanged.

### Everything replicates under a second algorithm

This is the point of running it. The deployable gap, independently reproduced:

| DEPLOYABLE − FULL, MAE | 20min | 1h | 3h | 6h | 12h | 24h | 48h |
|---|---|---|---|---|---|---|---|
| XGBoost | +1.27 | +1.09 | +0.85 | +0.35 | −0.21 | −0.17 | +0.15 |
| **LightGBM** | +1.34 | +1.27 | +0.46 | +0.26 | −0.47 | −0.19 | −0.16 |

Two unrelated tree implementations agree that satellite-only features are worth
~1.3 W/m² sub-hourly and nothing past 6 h. The headline Contribution 4 number is
now a replicated result rather than a single run.

### Finding: the NWP path has a measured ceiling, and both algorithms hit it

MAE improvement from adding known-future weather, mean over 7 sites, FULL:

| | 20min | 3h | 6h | 12h | 24h | 36h | 48h |
|---|---|---|---|---|---|---|---|
| **realistic** − nwp_free (XGB) | 0.06 | 1.40 | 2.46 | 4.97 | 6.92 | 7.34 | 8.35 |
| **realistic** − nwp_free (LGB) | 0.03 | 1.08 | 2.89 | 4.54 | 6.23 | 7.79 | 8.88 |
| **perfect** − nwp_free (XGB) | 0.06 | 3.55 | 6.75 | 11.02 | 13.66 | 15.75 | 16.81 |
| **perfect** − nwp_free (LGB) | 0.02 | 3.57 | 7.13 | 11.00 | 13.53 | 15.86 | 16.95 |

The two algorithms agree to within noise (±0.5 W/m²) on **how much information
a weather forecast contains**, which is what you would expect if this is a
property of the data rather than of the model.

Two numbers matter here:

1. **A realistic NWP forecast is worth ~6 W/m² at 24 h and ~9 W/m² at 48 h** —
   against a base MAE of ~106. Real, but 6–8%.
2. **A *perfect* forecast is worth ~13.5 W/m² at 24 h and ~17 W/m² at 48 h.** So
   the realistic track already captures about **half** the available NWP
   headroom (46% at 24 h, 52% at 48 h).

> **This sharpens the project's central hypothesis and partly tempers it.** The
> reasoning after the XGBoost grid was that the encoder is tapped out, so any
> architectural win has to come from the decoder/NWP path. That still holds —
> it is the only path with measurable headroom. But the headroom is now
> *quantified*, and it is bounded: even a perfect weather forecast, perfectly
> exploited, moves day-ahead MAE by ~13%. Half of that is already taken. So the
> realistic prize for Tropical-TFT's decoder work is single-digit W/m², and the
> thesis should say so before claiming it rather than after failing to find it.
>
> It is also a fourth independent line of evidence for the predictability
> ceiling, and the first one that isolates the *decoder* side specifically.

## The Loss Function Is the Biggest Lever Found (462/462)

Refitting the whole LightGBM grid with `--objective mae` — 462 paired cells,
every site, horizon, track and feature set:

| | squared error | **`mae`** | |
|---|---|---|---|
| Mean MAE | 94.16 | **89.54** | **+4.62 W/m², 462/462 wins, p = 2.0e-77** |
| Mean RMSE | 132.60 | 134.49 | −1.89 W/m², loses 420/462 |
| **vs smart persistence, both criteria** | 66/77 | **77/77** | |
| **vs NWP incumbent, regulated horizons** | 19/21 | **21/21** | margin +7.1 → **+11.6** |

The gain is flat across the entire horizon range, which is the surprising part:

| 20min | 30min | 1h | 2h | 3h | 6h | 12h | 24h | 48h |
|---|---|---|---|---|---|---|---|---|
| +4.96 | +4.82 | +4.57 | +4.86 | +4.77 | +4.26 | +4.41 | +4.72 | +4.20 |

The earlier XGBoost probe only looked sub-hourly and suggested this was a
sub-hourly patch. It is not. **Absolute error is the better objective for this
problem at every horizon tested.**

### It replicates across algorithms — 924/924 cells

The obvious objection is that this is an artefact of one library's L1
implementation. It is not. Both grids, full coverage:

| Algorithm | squared | **`mae`** | Gain | Wins | Wilcoxon | RMSE cost |
|---|---|---|---|---|---|---|
| LightGBM | 94.16 | **89.54** | **+4.62** | **462/462** | 2.0e-77 | 1.89 |
| XGBoost | 94.71 | **90.97** | **+3.73** | **462/462** | 2.0e-77 | 1.51 |

**Not one cell out of 924 goes the other way.** XGBoost's gain is smaller —
3.73 against 4.62 — so some of LightGBM's margin is implementation-specific,
but the bulk of the effect is not. Positive at all eleven horizons for both
(LightGBM +3.60 to +5.28, XGBoost +2.92 to +3.91).

The AEMO ladder against smart persistence, all 77 site-horizon pairs:

| Configuration | Both criteria |
|---|---|
| xgboost | 64/77 |
| xgboost-mae | 75/77 |
| lightgbm | 66/77 |
| **lightgbm-mae** | **77/77** |

`lightgbm-mae` is the only one of the four that clears every pair.
XGBoost-mae's two remaining failures are **Manila at 20 min and 30 min**, on
MAE alone — its RMSE beats persistence comfortably there (114.4 against 126.8).
Manila is the hardest site in the set, and it is precisely where the two
algorithms separate.

> **Why absolute error wins, physically.** Squared error fits the conditional
> *mean*. The tropical clear-sky-index distribution is skewed and heavy-tailed
> — long stretches near 1.0 punctuated by deep convective collapses — and the
> mean is a poor summary of a distribution shaped like that. Absolute error
> fits the conditional *median*, which sits where the mass actually is. That
> the effect survives two unrelated implementations is what makes this a
> property of the data rather than of a solver.

### Decision: report both objectives, do not freeze one

Rather than promote `mae` to the default the way `csi` was frozen as the
target, **both objectives are carried side by side in every results table**,
like the three NWP tracks. `config.DEFAULT` stays on squared error and the
absolute-error grid runs under the `-mae` label, so both are always present.

The reasoning is that the loss is not a nuisance parameter to be settled once.
It is a genuine operating choice with a measurable trade — roughly 4.6 W/m² of
MAE against 1.9 of RMSE and 0.011 of Forecast Skill — and which side of that
trade is right depends on what the forecast is scored against. A regulator
applying AEMO's dual criterion wants the absolute-error model. A comparison
quoted on Forecast Skill alone wants the squared-error one. Freezing either
would hide a choice the reader should see.

Coverage is complete for both algorithms: LightGBM and XGBoost each carry a
full grid under each objective, 3,234 rows apiece.

> [!warning] This decision has a Phase 4 consequence that must not be lost
> **Module D trains the quantile head with pinball loss, and pinball at the
> median is exactly `0.5·|error|` — absolute error.** So Tropical-TFT's P50
> forecast is trained on L1 by construction.
>
> Comparing it against squared-error trees would therefore confound the
> architecture with the loss function, and that confound is worth **4.6 W/m²**
> — larger than the entire realistic NWP headroom of ~7 W/m². A Tropical-TFT
> "win" of that size could be nothing but the objective.
>
> **The ablation must compare the TFT against `lightgbm-mae`, not `lightgbm`.**
> Quote the squared-error column too, but the like-for-like comparison is the
> absolute-error one.

### It also improves ramp detection, which is the opposite of the obvious worry

RMSE weights large errors, and the large errors here are convective ramps —
the events this project argues are operationally decisive. So trading RMSE for
MAE looked like it might quietly damage the thing that matters most. Measured
on KL, Manila and Bangkok at 20min/1h/2h:

| Metric | Δ (mae − squared) | Cells improved |
|---|---|---|
| **Recall** | **+0.042** | **9/9** |
| **F1** | **+0.021** | **9/9** |
| Precision | −0.010 | 3/9 |

KL at 20 min: recall 0.734 → **0.807**. Recall is the safety-critical number —
a missed down-ramp is unserved load or a reserve call — so this is the right
direction to move.

> **Why it goes this way.** Squared error drives predictions toward the
> conditional *mean*, which smooths; a smoothed forecast under-shoots sharp
> transitions and misses ramps. Absolute error targets the conditional
> *median*, which is far less pulled by the tails and keeps transitions sharp.
> A sharper forecast calls more ramps. The small precision cost is the
> expected other side of the same mechanism.

### Two mechanisms to state honestly in the write-up

**Early stopping follows the objective.** Neither library is given an explicit
`eval_metric`, so it defaults to the training loss. The `mae` models stop on
validation L1, the squared models on validation L2. Part of the gain is
therefore *model selection*, not only the fitting loss. Both configurations are
internally consistent and either is what one would deploy, so the honest claim
is "a model fitted **and selected** for absolute error beats one fitted and
selected for squared error" — not "one loss term is worth 4.6 W/m²".

**Training space and scoring space differ.** The model trains on the clear-sky
index but is scored on `GHI = csi × clearsky`. Squared error in CSI space is
not squared error in GHI space — the clear-sky multiplier reweights everything
toward midday. Absolute CSI error sits closer to GHI MAE than squared CSI error
does, which is part of why the gain is so uniform.

Neither mechanism undermines the result. Both explain it.

## Two Levers That Did Nothing (useful negatives)

### Pooled multi-site training: +0.32 W/m²

One model over all seven sites with a `site_id` feature, **522,152 training
rows against ~75,000 per site**, evaluated per site on its own test rows:

| Horizon | Mean gain |
|---|---|
| 20min | +0.49 |
| 1h | +0.53 |
| 6h | −0.18 |
| 24h | +0.45 |

**Seven times the data buys 0.4%**, with signs flipping site to site. This is
the strongest available evidence that **data volume is not the constraint**.

One real signal inside the noise: **Ho Chi Minh gains at every horizon** (+1.18
to +2.41) while Kota Kinabalu and Penang consistently lose. Pooling helps some
sites and hurts others, which is a preliminary answer to the transfer question
the zero-shot and few-shot studies are designed to settle.

### Short-window variability features: +0.03 W/m²

The feature set carries exactly one rolling standard deviation, at 6 h, yet
site difficulty correlates r = +0.854 with CSI interquartile range. Adding
30 min, 1 h and 3 h windows (82 → 88 columns), tested at 20min/30min/1h/3h on
KL, Manila and Bangkok: **mean MAE gain +0.033 W/m², RMSE +0.129**. Noise, with
signs flipping cell to cell.

This was the most physically motivated feature bet available — aimed by the
project's own variability finding at the one regime where features still
measurably matter — and it returned nothing.

> **The pattern across seven experiments.** Every attempt to give the model
> *more information* has failed: more features, more data, more capacity,
> deeper trees, a different algorithm, targeted variability features. The only
> intervention that worked changed what the model is asked to *optimise*, not
> what it knows. That is the predictability-ceiling result stated precisely,
> and it is now the best-evidenced claim in the project.

## Operational NWP Baseline (the incumbent comparison)

`scripts/run_nwp_baseline.py` — 9,702 rows, 29s, all seven sites.
Module: `src/solarfc/nwp_baseline.py`. 28 tests.

This is the baseline that speaks to adoption. Every other reference here is
statistical or machine-learned; this one is the forecast a plant already
receives from a national met agency, and AEMO accredits a self-forecast only
if it beats that on **both** MAE and RMSE.

> **It is a cloud-driven NWP baseline, not an operational GHI forecast.** JMA
> GSM carries no shortwave radiation, so irradiance is derived from archived
> forecast *cloud* through a clear-sky transmittance model. The cloud is a
> genuine forecast at a genuine lead time; the conversion is ours. Label it
> that way every time it is reported.

Three variants, all fitted on training years only (the archive starts in
2018, so in practice they fit on 2018 and are scored on 2019 and 2020):

| Model | What it is |
|---|---|
| `nwp_jma_mos` | Ridge post-processing over all five forecast fields — **the headline incumbent** |
| `nwp_jma_cloud` | Fitted cloud-to-transmittance curve |
| `nwp_jma_cloud_kc` | The same curve with Kasten-Czeplak's published coefficients |

MOS is the headline because raw NWP output is post-processed before anyone
dispatches on it. It roughly doubles explained variance (R² 0.05 → 0.15 at
KL), so scoring only the cloud-only form would understate the incumbent and
flatter everything measured against it.

### The headline: the model beats the incumbent

AEMO dual criterion at the three regulated horizons (24h/36h/48h × 7 sites
= 21 pairs), `realistic`/FULL/`csi` against `nwp_jma_mos` at its
operational lead:

| Model | MAE | RMSE | **Both** | Mean margin |
|---|---|---|---|---|
| XGBoost | 19/21 | 21/21 | **19/21** | +6.5 W/m² |
| LightGBM | 19/21 | 21/21 | **19/21** | **+7.1 W/m²** |

Per site at 24 h, MAE:

| Site | LightGBM | NWP MOS | NWP cloud | Smart pers. | Margin |
|---|---|---|---|---|---|
| bangkok | 85.5 | 100.1 | 100.9 | 108.7 | **+14.5** |
| manila | 125.7 | 136.9 | 136.8 | 155.6 | +11.2 |
| penang | 105.8 | 115.5 | 116.4 | 145.3 | +9.8 |
| kota_kinabalu | 105.1 | 112.9 | 120.9 | 131.6 | +7.8 |
| ho_chi_minh | 102.4 | 109.5 | 110.9 | 126.5 | +7.0 |
| kuala_lumpur | 108.0 | 113.4 | 118.6 | 148.0 | +5.4 |
| **jakarta** | 112.8 | **110.7** | 122.4 | 146.8 | **−2.2** |

Both failures are Jakarta, at 24 h and 36 h, and **both are MAE-only — RMSE
passes at each**. That is the same squared-loss trade-off found sub-hourly
against persistence, so the `--objective mae` fix is the thing to try.

### Finding: the bottleneck is the conversion, not the forecast

Scoring at lead 0 (JMA's analysis-time cloud — not a forecast, so a ceiling)
against lead 24 h isolates how much error comes from the forecast decaying
rather than from everything else:

| Site | lead 0 (analysis) | lead 24 h (forecast) | Cost of forecasting |
|---|---|---|---|
| bangkok | 100.1 | 100.1 | **0.0** |
| kota_kinabalu | 112.1 | 112.9 | 0.7 |
| kuala_lumpur | 111.5 | 113.4 | 1.9 |
| jakarta | 108.6 | 110.7 | 2.1 |
| ho_chi_minh | 106.6 | 109.5 | 2.9 |
| penang | 108.3 | 115.5 | 7.3 |
| manila | 128.3 | 136.9 | 8.6 |

**Mean cost of a 24-hour lead: 3.4 W/m², against a total error of ~113.**
So ~97% of the incumbent's error is in the cloud-to-irradiance conversion
and the coarse grid, not in the cloud forecast going stale. Consistent with
the earlier finding that JMA's forecast decay is modest and most of the
apparent disagreement with ERA5 is inter-model representation.

### Kasten-Czeplak does not transfer, and fitting it halved our own margin

The published curve `csi = 1 − 0.75·cc^3.4` predicts a clear-sky index of
**0.25** at full overcast. These seven sites measure **0.63**. Scored on the
test year it gives a *negative* R² at every site — worse than predicting the
training mean. Fitted coefficients land near **a = 0.37, b = 0.72**.

| Site | Fitted MAE | Kasten-Czeplak MAE | Penalty |
|---|---|---|---|
| kuala_lumpur | 118.6 | 139.8 | 21.3 |
| penang | 116.4 | 135.6 | 19.2 |
| jakarta | 122.4 | 137.0 | 14.6 |
| ho_chi_minh | 110.9 | 121.8 | 10.9 |
| kota_kinabalu | 120.9 | 131.2 | 10.3 |
| manila | 136.8 | 146.1 | 9.3 |
| bangkok | 100.9 | 109.4 | 8.4 |

> **Worth stating plainly in Chapter 5.** Using the published coefficients
> would have made the incumbent look ~13 W/m² worse than it is — roughly
> twice the true margin. Fitting the baseline properly *halved this
> project's own headline number*. That is the difference between a real
> result and an artefact of a badly specified comparator.

### Why cloud cover is such a weak predictor here

Even fitted, cloud cover explains little: R² 0.04–0.14 across the sites. Total
cloud fraction over a coarse grid cell is a poor proxy for point transmittance
in the tropics, where thin cirrus is common and the cloud-cover distribution is
compressed against its upper bound.

The weakness is not uniform, and **the latitude gradient reappears by a
completely independent route**:

| Group | Sites | r (cloud vs CSI) | R² |
|---|---|---|---|
| Equatorial (\|lat\| < 7) | KL, Penang, Kota Kinabalu | −0.22 to −0.24 | 0.04–0.05 |
| Off-equator | Ho Chi Minh, Bangkok, Manila | −0.36 to −0.42 | 0.11–0.14 |

Same direction as the r = 0.834 correlation between |latitude| and JMA
forecast skill, but derived from cloud physics rather than forecast
verification. Two independent measurements of the same thing.

### Reading the horizon axis

A forecast valid at time T is a single value however far ahead it was issued,
so the NWP prediction does not change across horizons at a fixed lead — only
the persistence reference it is scored against does. That asymmetry is exactly
why NWP looks progressively better as the horizon grows.

The archive serves fixed daily offsets, so there is nothing between lead 0 and
lead 24 h. Sub-daily horizons are therefore mapped to the 24-hour lead, which
badly understates what a fresh run would give: at 20 min the incumbent scores
113 against smart persistence at 64. **Do not report the short-horizon NWP
comparison as meaningful.** The mapping is exact only at 24 h, 36 h and 48 h —
which are precisely the horizons the grid code regulates.

## Predictability Study — TIGGE Ensemble

`scripts/pull_tigge_ensemble.py`. Download in progress; no results yet.

### Why this exists

Seven experiments now say the tree models sit at a predictability ceiling
rather than a capacity or feature one. That is a claim about the
*atmosphere*, and it deserves to be measured directly rather than inferred
from things that failed to help.

Yang's framework does exactly that. Predictability is bounded between a
**highest tolerable MSE** — derived from a correlogram fitted to the lag-h
autocorrelation of the clear-sky index, referenced to the optimal convex
combination of climatology and persistence — and a **smallest attainable
MSE**, approximated by predictability error growth: how fast a control
forecast and its perturbed siblings diverge in a real dynamical model.

The upper bound needs nothing but data already on disk. The lower bound
needs an NWP ensemble, which is what this downloads.

| Source | Yang (2022) *RSER* 167:112736; Yang et al. (2023) *Solar Energy*; Liu & Yang (2023) *RSER* 182 |
|---|---|
| Their empirics | 7 sites in the United States, and a CONUS map |
| This project | 7 sites in equatorial SEA, monsoon-stratified |

An OpenAlex sweep found **8 works total** at the intersection of
predictability and solar forecast skill, and **4** for monsoon plus solar
predictability, none of them relevant. The method is established; the
region is unoccupied.

### What is being pulled

| | |
|---|---|
| Dataset | `tigge-forecasts` on the ECMWF Data Store, `origin=ecmwf` |
| Variable | `surface_net_solar_radiation` (`ssr`), accumulated J m⁻² |
| Members | control + all 50 perturbed — the form has no member selector |
| Leads | 6-hourly to 48 h, then daily to 168 h (14 values) |
| Period | 2020, one 00:00 UTC initialisation per day |
| Volume | ~13.6 GB raw, under 100 MB extracted |

Lead times run past the 48 h horizon ceiling on purpose. Two leads would
give Yang's bound at 24 h and 48 h only; the full curve is what supports a
statement about an equatorial predictability *horizon* — where skill
saturates — which is the more quotable result and ties directly to the
measured latitude gradient.

### Things that cost time to find out

- **ECDS runs one job at a time per user.** Parallel processes only queue
  behind each other. Measured: a perturbed month is ~1109 MB and 25–40 min,
  a control month ~21 MB and 90 s. Whole year is 5–8 h, almost all queue
- **Credentials go in `.env` as `ECDS_TOKEN`, never `~/.cdsapirc`.** ECDS
  wants the same filename as the Copernicus CDS with a different url and
  key, so writing it there silently breaks the ERA5 pipeline
- The old `ecmwf-api-client` / `.ecmwfapirc` route was **decommissioned on
  27 May 2026**; TIGGE moved to ECDS on 21 April. Ignore any guide older
  than that. ECMWF registration is Keycloak SSO — the Drupal
  `/user/register` form is vestigial and its username field is dead
- TIGGE is **GRIB only**, so this is the one part of the pipeline needing
  `cfgrib` and `eccodes`; everything else is NetCDF or Parquet
- The grid is **reduced Gaussian at ~0.14°**, so latitude and longitude
  arrive as flat arrays over a `values` dimension and nearest-neighbour
  search has to walk them directly. Upside: the closest point to KL is
  **4.5 km** away, against ERA5's 31 km
- `ssr` accumulates from forecast start and the lead spacing is uneven, so
  the divisor must come from the actual step difference
- Extraction uses `isel(values=i).to_dataframe()` rather than positional
  numpy indexing: a monthly file carries `time` as a dimension where a
  single-day one has it as a scalar

> **Licence: CC BY 4.0, not non-commercial.** TIGGE splits by producing
> centre — BoM, CMA, CPTEC, IMD, JMA, MF and NCMRWF are CC BY-NC 4.0, but
> DWD, ECCC, **ECMWF**, KMA, NCEP and UKMO are plain CC BY 4.0. Since
> `origin=ecmwf`, this is redistributable on the same terms as ERA5 and can
> go in the benchmark deposit. Published work must acknowledge TIGGE.
> **Do not widen `origin` to another centre** without re-checking — that
> would pull NC data in and force the whole deposit to non-commercial.

### Validated before the full run

One day at KL, 2020-06-15: control gives 158.6 W/m² over the 24–48 h
interval, and the 50-member ensemble spans **99.3 to 184.9 W/m²** with a
mean of 158.1 — the control sitting essentially on the ensemble mean. That
spread is the predictability signal the lower bound is built from.

Cross-check on the field itself: 0–24 h accumulates 15.38 MJ/m², and KL runs
about 18 MJ/m²/day of GHI, which at `(1 − albedo) ≈ 0.87` predicts 15.7. The
retrieval is what it claims to be.

`ssr` is *net*, not global. The conversion to GHI is left to the analysis
step, where NSRDB's per-timestep `surface_albedo` beats any constant this
script could assume — and for a control-versus-perturbed spread the albedo
factor very nearly cancels anyway.

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
   operational GHI forecast. **Built** — see the Operational NWP Baseline
   section above for the results and their caveats.
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

## Redistribution Licensing — settled

Checked because it decides what the benchmark release ships: the derived
dataset itself, or derivation scripts plus checksums. Answer: it can ship the
data, with one exclusion.

| Source | Licence | Derived data redistributable? |
|---|---|---|
| **NSRDB** | [CC BY 3.0 US](https://registry.opendata.aws/nrel-pds-nsrdb/) (AWS Open Data registry) | **Yes**, with attribution |
| **ERA5** | [CC BY 4.0](https://forum.ecmwf.int/t/cc-by-licence-to-replace-licence-to-use-copernicus-products-on-02-july-2025/13464) since 2 July 2025 | **Yes**, with attribution |
| **JMA GSM** via Open-Meteo | CC BY 4.0 | **Yes**, with attribution |
| **Solcast** | [Researcher account terms](https://solcast.com/solar-data-api/api/terms-and-conditions/) | **No** — no distribution to third parties, directly or bundled into another product, without written permission |

ERA5 was the one at real risk and it resolved the easy way: the old bespoke
"Licence to use Copernicus Products" was replaced by plain CC-BY 4.0 in July
2025, so derived redistribution is explicitly fine.

**What the release ships:** the preprocessed 10-minute aligned NSRDB + ERA5
dataset, the frozen split indices, the evaluation harness and every baseline
result. **Solcast values are excluded** — the spot-validation appears as
aggregate comparison statistics only, which is use rather than redistribution.

Required attribution in the deposit:

- NSRDB — cite Sengupta et al. (2018), *Renewable and Sustainable Energy
  Reviews* 89:51–60, and credit DOE/NREL/ALLIANCE
- ERA5 — "Generated using Copernicus Climate Change Service information [year]"
- Open-Meteo — CC BY 4.0 attribution

> **Solcast has a pre-publication obligation, and it has a deadline.** The
> researcher terms require sending Solcast a pre-print of any thesis,
> presentation or publication using their data **at least one week before**
> dissemination. That applies to the thesis submission itself. Diarise it —
> it is easy to miss and it is a term of the account, not a courtesy.

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
- [x] **Full XGBoost grid run** — 924 models, 0 failures, `csi` target frozen
- [x] **LightGBM on the winning target** — 444 models, 0 failures, 18m34s. Wins
      395/462 on MAE by 0.55 W/m². Every Phase 2 finding replicates
- [x] **Operational NWP baseline** — cloud-driven, three variants, 9,702 rows.
      Model beats the MOS incumbent 19/21 at the regulated horizons, +7.1 W/m²
- [ ] LSTM encoder-decoder, BiLSTM-GRU, physics-guided CNN-BiLSTM (MIMO)
- [ ] Chronos-2 zero-shot baseline
- [ ] Cloud-driven NWP baseline for the 2020 benchmark
- [x] Dataset redistribution licence check — the release ships data, Solcast excluded

---

## Resume Here

**State as of 2026-08-31: Phase 2 is complete.** XGBoost and LightGBM grids
run, the operational NWP incumbent is built and scored, and the
redistribution licence question is settled. Next phase is Standard TFT and
the Transformer baselines.

### Run this first

```bash
/opt/anaconda3/envs/fyp/bin/python -m pytest      # expect 265 passing
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

# 5b. Same grid under absolute error -- both objectives are reported,
#     so this is not optional. ~27 min (L1 is slower than L2).
python scripts/train_gbdt.py --algorithm lightgbm --targets csi \
    --objective mae --label mae

# 6. Operational NWP incumbent -- ~30s. Needs the JMA cache.
python scripts/run_nwp_baseline.py

# 7. TIGGE ensemble for the predictability lower bound. ~13.6 GB and
#    5-8 h, so run it on a machine that stays awake. Resumable.
python scripts/pull_tigge_ensemble.py
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
| Evaluation framework | 265 tests, metrics frozen before any model |
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
- **Merge to `main` and tag `phase-2-baselines`** — the phase is complete, and
  the repo convention is to tag at phase completion
- **Jakarta fails AEMO against the NWP incumbent** at 24 h and 36 h, on MAE
  only. Try `--objective mae`, the same fix that worked sub-hourly
- **Solcast pre-print obligation** — a pre-print must reach Solcast at least a
  week before the thesis is submitted. See the Redistribution Licensing section
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
