"""Project-wide constants: sites, horizons, monsoon phases, splits, paths.

Everything downstream imports from here so that a change to the horizon set or
the split boundaries propagates to every model and every results table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
NSRDB_DIR = DATA_DIR / "nsrdb"
ERA5_DIR = DATA_DIR / "era5"
SOLCAST_DIR = DATA_DIR / "solcast"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = PROCESSED_DIR / "figures"
RESULTS_DIR = PROCESSED_DIR / "results"

# --------------------------------------------------------------------------
# Temporal resolution
# --------------------------------------------------------------------------

#: Native NSRDB Himawari PSM v3 resolution.
STEP_MINUTES = 10

# --------------------------------------------------------------------------
# Forecast horizons
# --------------------------------------------------------------------------
#
# Horizons are defined in STEPS, not minutes, because only integer multiples of
# STEP_MINUTES are representable on the NSRDB grid.
#
# NOTE (2026-08-30): the plan originally listed a 15-minute horizon. That is not
# representable at 10-minute resolution (1.5 steps), so it is replaced by 20
# minutes (2 steps). 20 min also sits *above* the 15-minute boundary below which
# the plan explicitly scopes out true nowcasting as requiring sky-camera or
# real-time optical-flow input, so the scope statement remains consistent.

HORIZON_STEPS: tuple[int, ...] = (2, 3, 6, 12, 18, 36, 72, 108, 144, 216, 288)

HORIZON_LABELS: tuple[str, ...] = (
    "20min",
    "30min",
    "1h",
    "2h",
    "3h",
    "6h",
    "12h",
    "18h",
    "24h",
    "36h",
    "48h",
)

assert len(HORIZON_STEPS) == len(HORIZON_LABELS)

#: Horizons required by the Malaysian LSS grid code, for the operational tables.
#:
#: The Rolling 24 Hours Forecast is a 24 h submission at 15-minute intervals,
#: re-issued every half hour. The Declared Daily Capacity is submitted day-ahead
#: by 10:00 and must cover to the end of the following day, which is 38 h out --
#: hence 36 h and 48 h, which the original 24 h ceiling could not reach.
REGULATORY_HORIZON_LABELS: tuple[str, ...] = ("24h", "36h", "48h")

#: Reporting interval mandated by the Energy Commission for LSS submissions.
#: Training runs on the native 10-minute grid because averaging to 15 minutes
#: destroys ramp structure, which is a headline metric here; predictions are
#: aggregated to 15 minutes for the operational tables.
REPORTING_STEP_MINUTES: tuple[int, ...] = (10, 15)

#: Minimum encoder lookback: 72 steps = 12 h. Abdul Rahman et al. (2026) find
#: >=12 h optimal for Malaysian solar data.
LOOKBACK_STEPS = 72

#: PatchTST / iTransformer are additionally evaluated at their near-optimal
#: lookback (288 steps = 48 h) — see the lookback fairness note in the plan.
LOOKBACK_STEPS_LONG = 288

# --------------------------------------------------------------------------
# Quantiles
# --------------------------------------------------------------------------

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

# --------------------------------------------------------------------------
# Sites
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    key: str
    label: str
    country: str
    latitude: float
    longitude: float
    #: Elevation in metres, as reported in the NSRDB file metadata.
    elevation: float
    #: Olson timezone, used only for local-time plots. All modelling is in UTC.
    timezone: str


SITES: tuple[Site, ...] = (
    Site("kuala_lumpur", "Kuala Lumpur", "Malaysia", 3.139, 101.687, 49, "Asia/Kuala_Lumpur"),
    Site("penang", "Penang", "Malaysia", 5.414, 100.330, 3, "Asia/Kuala_Lumpur"),
    Site("kota_kinabalu", "Kota Kinabalu", "Malaysia", 5.980, 116.073, 3, "Asia/Kuching"),
    Site("ho_chi_minh", "Ho Chi Minh City", "Vietnam", 10.823, 106.630, 9, "Asia/Ho_Chi_Minh"),
    Site("bangkok", "Bangkok", "Thailand", 13.754, 100.501, 4, "Asia/Bangkok"),
    Site("jakarta", "Jakarta", "Indonesia", -6.208, 106.846, 8, "Asia/Jakarta"),
    Site("manila", "Manila", "Philippines", 14.599, 120.984, 16, "Asia/Manila"),
)

SITES_BY_KEY: dict[str, Site] = {s.key: s for s in SITES}

SITE_KEYS: tuple[str, ...] = tuple(s.key for s in SITES)

#: Used by the generalisation test: train on these, evaluate on the rest.
MALAYSIAN_SITE_KEYS: tuple[str, ...] = ("kuala_lumpur", "penang", "kota_kinabalu")

#: NSRDB coastal-site caveat: reported satellite bias is materially larger at
#: coastal sites, which is most of this benchmark. Absolute errors at these
#: sites carry a data-quality floor that is not model error. See plan, Datasets.
COASTAL_SITE_KEYS: tuple[str, ...] = (
    "penang",
    "kota_kinabalu",
    "ho_chi_minh",
    "jakarta",
    "manila",
)

# --------------------------------------------------------------------------
# Monsoon phases
# --------------------------------------------------------------------------
#
# Malaysian Meteorological Department convention. Month-based labelling is a
# deliberate simplification: true onset shifts year to year, which is exactly
# what the Module A gate-validation study measures the learned gate against.

NE_MONSOON = 0
INTER_MONSOON_I = 1
SW_MONSOON = 2
INTER_MONSOON_II = 3

MONSOON_LABELS: dict[int, str] = {
    NE_MONSOON: "NE Monsoon",
    INTER_MONSOON_I: "Inter-monsoon I",
    SW_MONSOON: "SW Monsoon",
    INTER_MONSOON_II: "Inter-monsoon II",
}

#: Month (1-12) -> phase code.
MONTH_TO_MONSOON: dict[int, int] = {
    1: NE_MONSOON,
    2: NE_MONSOON,
    3: NE_MONSOON,
    4: INTER_MONSOON_I,
    5: SW_MONSOON,
    6: SW_MONSOON,
    7: SW_MONSOON,
    8: SW_MONSOON,
    9: SW_MONSOON,
    10: INTER_MONSOON_II,
    11: NE_MONSOON,
    12: NE_MONSOON,
}

#: Nominal monsoon transition dates as (month, day), used to build the
#: transition windows over which quantile calibration is reported separately.
#: Calibration typically degrades under regime shift; reporting a single
#: aggregate PICP would hide exactly that failure.
MONSOON_TRANSITIONS: tuple[tuple[int, int], ...] = (
    (4, 1),   # NE -> Inter-monsoon I
    (5, 1),   # Inter-monsoon I -> SW
    (10, 1),  # SW -> Inter-monsoon II
    (11, 1),  # Inter-monsoon II -> NE
)

#: Half-width of a transition window, in days.
#:
#: Chosen at 10 rather than the initially drafted 21. At +/-21 days the four
#: boundaries cover ~40% of the year — the Apr 1 and May 1 windows merge, making
#: the whole of April transitional — which destroys the contrast the stratum
#: exists to provide. At +/-10 days the windows are disjoint and cover ~22%,
#: a genuine minority stratum against which stable-period calibration can be
#: compared. Note that the inter-monsoon phases are themselves transitional by
#: definition and remain separately visible via ``monsoon_phase``.
TRANSITION_WINDOW_DAYS = 10

# --------------------------------------------------------------------------
# Temporal splits — strictly chronological, never shuffled
# --------------------------------------------------------------------------

TRAIN_YEARS: tuple[int, ...] = (2016, 2017, 2018)
VAL_YEARS: tuple[int, ...] = (2019,)
TEST_YEARS: tuple[int, ...] = (2020,)

ALL_YEARS: tuple[int, ...] = TRAIN_YEARS + VAL_YEARS + TEST_YEARS

# --------------------------------------------------------------------------
# Evaluation masking
# --------------------------------------------------------------------------
#
# Night-time zeros are trivially predictable and would flatter every metric:
# they shrink MAE, inflate R², and make MAPE undefined. All headline metrics are
# therefore computed on daytime samples only, defined by a clear-sky floor.

#: Samples with clear-sky GHI at or below this (W/m^2) are excluded as night.
DAYTIME_CLEARSKY_FLOOR = 20.0

#: Additional floor for MAPE only, which is unstable as the denominator -> 0.
#: Prefer the normalised metrics (nMAE / nRMSE) over MAPE when reporting.
MAPE_GHI_FLOOR = 50.0

# --------------------------------------------------------------------------
# Ramp events
# --------------------------------------------------------------------------
#
# The operationally decisive metric. Ramps are rare, so RMSE is dominated by
# unremarkable timesteps and can look excellent while every ramp is missed.

#: A ramp occurs when |dGHI| over the window exceeds this fraction of the
#: concurrent clear-sky GHI.
RAMP_THRESHOLD_FRAC = 0.50

#: Window over which the change is measured, in steps (3 steps = 30 min).
RAMP_WINDOW_STEPS = 3

#: Swept in the sensitivity analysis so the headline threshold is justified
#: rather than asserted.
RAMP_THRESHOLD_SWEEP: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)
RAMP_WINDOW_SWEEP: tuple[int, ...] = (1, 3, 6)

# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------
#
# NSRDB supplies aerosol optical depth, ozone, an asymmetry parameter and a
# cloud-type classification. No PV plant can measure any of these. A model that
# depends on them scores well on this benchmark and cannot run on a real site,
# and nothing in an accuracy table would reveal that -- so the deployable set is
# a reported variant, not a footnote.
#
# DEPLOYABLE covers what IEC 61724-1 instrumentation actually provides. Under
# the Malaysian LSS rules a site has at least one pyranometer and one full
# weather station per 10 MWac (per 1 MW if distribution-connected), logging at
# 15-minute resolution or better.

#: NSRDB columns unavailable to any real plant.
SATELLITE_ONLY_FEATURES: tuple[str, ...] = (
    "Aerosol Optical Depth",
    "Alpha",
    "Ozone",
    "Asymmetry",
    "Cloud Type",
    "Surface Albedo",
)

#: Measurable on site with IEC 61724-compliant instrumentation.
DEPLOYABLE_NSRDB_FEATURES: tuple[str, ...] = (
    "GHI",
    "DNI",
    "DHI",
    "Clearsky GHI",
    "Temperature",
    "Dew Point",
    "Relative Humidity",
    "Pressure",
    "Wind Speed",
    "Wind Direction",
    "Solar Zenith Angle",
)
