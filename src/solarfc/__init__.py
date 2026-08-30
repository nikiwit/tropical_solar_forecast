"""solarfc — evaluation and baseline framework for the Tropical-TFT project.

The evaluation framework is built and frozen before any model is trained, so
that every baseline, every ablation variant and the C++ engine are all scored
by identical code.

Modules
-------
config      Sites, horizons, monsoon phases, split years, thresholds.
data        NSRDB Himawari loading and grid-continuity checks.
era5        ERA5 loading, site extraction and upsampling to the NSRDB grid.
splits      Chronological splits, monsoon labelling, transition windows.
metrics     Point, skill and probabilistic metrics.
ramp        Ramp-event detection and scoring.
baselines   Naive persistence, smart persistence, clear-sky.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import baselines, config, data, era5, metrics, ramp, splits

__all__ = [
    "baselines",
    "config",
    "data",
    "era5",
    "metrics",
    "ramp",
    "splits",
    "__version__",
]
