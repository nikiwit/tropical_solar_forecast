"""One results schema, written to by every model.

Why a schema rather than per-experiment tables
----------------------------------------------
Baselines, gradient boosting, the Transformer models, the module ablation,
the multi-site and few-shot studies and the deployment benchmark all emit
results. If each writes its own table shape, every comparison across them in
Chapter 5 becomes a manual join, and the ablation table -- the one that decides
whether the contributions hold -- gets assembled by hand at exactly the point
where a mistake is least visible.

So every model appends rows in one long format:

    one row = one (model, site, horizon, track, feature_set, target,
    split,
                   stratum) combination, with its metrics

Stratification is a column rather than a separate file. The
monsoon-phase tables, the transition-vs-stable calibration split and the
aggregate all come out of the same frame by filtering, which is what
keeps them consistent with each other.

Everything needed to identify a run travels with the row. A results file
that cannot say which code produced it is not reproducible, and by
Chapter 5 there will be thousands of rows from months of runs.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .config import (
    DAYTIME_CLEARSKY_FLOOR,
    MONSOON_LABELS,
    RESULTS_DIR,
    TRANSITION_WINDOW_DAYS,
)
from .splits import is_transition_window, monsoon_phase

__all__ = [
    "RESULT_COLUMNS",
    "RunMeta",
    "score_predictions",
    "append_results",
    "load_results",
    "pivot_horizons",
]

#: Identity of a row, before the metric columns.
KEY_COLUMNS: tuple[str, ...] = (
    "model",
    "site",
    "horizon_label",
    "horizon_steps",
    "track",
    "feature_set",
    "target",
    "split",
    "stratum",
    "stratum_kind",
)

#: Metric columns. Fixed order so files concatenate cleanly across
#: phases.
METRIC_COLUMNS: tuple[str, ...] = (
    "n",
    "mae",
    "rmse",
    "mbe",
    "mape",
    "nmae",
    "nrmse",
    "r2",
    "fs_smart",
    "fs_naive",
)

RESULT_COLUMNS: tuple[str, ...] = (
    KEY_COLUMNS
    + METRIC_COLUMNS
    + (
        "run_id",
        "timestamp",
        "git_commit",
    )
)


def _git_commit() -> str:
    """Short commit hash, or 'unknown' outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class RunMeta:
    """Provenance travelling with every row of a run.

    ``git_commit`` is the one that matters. Thousands of rows will
    accumulate over the project, and without it a surprising number in
    Chapter 5 cannot be traced to the code that produced it.
    """

    run_id: str
    model: str
    notes: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    git_commit: str = field(default_factory=_git_commit)
    python: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    hyperparameters: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _strata(index: pd.DatetimeIndex) -> list[tuple[str, str, np.ndarray]]:
    """Every stratum a row can be reported under, as (kind, name, mask).

    ``all`` is included so the aggregate and the stratified numbers are
    produced by the same code path. A separately-computed aggregate is
    how an aggregate stops matching its own breakdown.
    """
    out: list[tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones(len(index), dtype=bool))
    ]

    phase = monsoon_phase(index)
    for code, label in MONSOON_LABELS.items():
        mask = phase == code
        if mask.any():
            out.append(("monsoon", label, mask))

    transition = np.asarray(
        is_transition_window(index, window_days=TRANSITION_WINDOW_DAYS),
        dtype=bool,
    )
    if transition.any():
        out.append(("regime", "transition", transition))
    if (~transition).any():
        out.append(("regime", "stable", ~transition))

    return out


def score_predictions(
    y_true,
    y_pred,
    clearsky_ghi,
    index: pd.DatetimeIndex,
    *,
    reference_smart=None,
    reference_naive=None,
    daytime_floor: float = DAYTIME_CLEARSKY_FLOOR,
    stratify: bool = True,
) -> pd.DataFrame:
    """Score one prediction series into schema rows, one per stratum.

    All inputs are GHI in W/m^2 at the target timestamp -- convert with
    :func:`solarfc.dataset.to_ghi` first, whatever the model was trained
    on.

    The daytime mask is applied once, here, and identically for every
    model and every reference. A metric that changes its own sample set
    is not comparable, and Forecast Skill computed against a reference
    scored on a different set is worse than no skill score at all.

    Parameters
    ----------
    reference_smart, reference_naive : array-like, optional
        Baseline predictions on the same index. Smart persistence is the
        primary Forecast Skill reference; naive is reported only for
        comparability with papers that use it.
    """
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    envelope = np.asarray(clearsky_ghi, dtype=float).ravel()
    index = pd.DatetimeIndex(index)

    if not (truth.shape == pred.shape == envelope.shape == (len(index),)):
        raise ValueError(
            f"shape mismatch: y_true {truth.shape}, y_pred {pred.shape}, "
            f"clearsky {envelope.shape}, index {(len(index),)}"
        )

    daytime = M.daytime_mask(envelope, floor=daytime_floor)
    smart = (
        None
        if reference_smart is None
        else np.asarray(reference_smart, float).ravel()
    )
    naive = (
        None
        if reference_naive is None
        else np.asarray(reference_naive, float).ravel()
    )

    strata = (
        _strata(index)
        if stratify
        else [("all", "all", np.ones(len(index), bool))]
    )

    rows = []
    for kind, name, stratum in strata:
        mask = daytime & stratum
        # Every model shares this mask, so a stratum with too few
        # samples is dropped consistently rather than producing a metric
        # from a handful.
        if mask.sum() < 30:
            continue

        row = {"stratum": name, "stratum_kind": kind, "n": int(mask.sum())}
        row.update(M.point_metrics(truth[mask], pred[mask]))
        row["fs_smart"] = (
            M.forecast_skill(truth[mask], pred[mask], smart[mask])
            if smart is not None
            else np.nan
        )
        row["fs_naive"] = (
            M.forecast_skill(truth[mask], pred[mask], naive[mask])
            if naive is not None
            else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows)


def append_results(
    frame: pd.DataFrame,
    path: str | Path | None = None,
    *,
    meta: RunMeta | None = None,
    **keys,
) -> Path:
    """Append scored rows to the results file, creating it if absent.

    CSV rather than Parquet: the file is read by hand constantly while
    writing up, diffs legibly in git, and will not outgrow it --
    thousands of rows, not millions.
    """
    path = Path(RESULTS_DIR / "results.csv" if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = frame.copy()
    for key, value in keys.items():
        out[key] = value
    if meta is not None:
        out["model"] = meta.model
        out["run_id"] = meta.run_id
        out["timestamp"] = meta.timestamp
        out["git_commit"] = meta.git_commit

    for column in RESULT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out = out[list(RESULT_COLUMNS)]

    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """Read the results file."""
    path = Path(RESULTS_DIR / "results.csv" if path is None else path)
    if not path.exists():
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    return pd.read_csv(path)


def pivot_horizons(
    frame: pd.DataFrame,
    metric: str = "mae",
    *,
    split: str = "test",
    stratum: str = "all",
    index=("model", "track", "feature_set"),
) -> pd.DataFrame:
    """Horizons across the columns -- the shape every results table wants.

    Rows are ordered by ``config.HORIZON_LABELS`` rather than
    alphabetically, so '2h' does not sort between '18h' and '20min'.
    """
    from .config import HORIZON_LABELS

    subset = frame[(frame["split"] == split) & (frame["stratum"] == stratum)]
    table = subset.pivot_table(
        index=list(index),
        columns="horizon_label",
        values=metric,
        aggfunc="mean",
    )
    ordered = [h for h in HORIZON_LABELS if h in table.columns]
    return table[ordered]
