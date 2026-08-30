"""Per-site z-score standardisation, fitted on the training years only.

Why per-site
------------
Each site gets its own mean and standard deviation, fitted on 2016-2018 and
frozen. Three reasons, in order of weight:

1. It matches the deployment story. A plant normalises against its own history,
   not against a pool that includes Bangkok and Jakarta. Contribution 4
   is a single-site GHI-to-MWac chain, so a single-site statistic is the
   honest one.
2. It keeps the SolarInfer equivalence check well posed. The C++ engine loads
   one site's parameters and must match PyTorch to 4 decimal places *in
   this space*, so the space has to be unambiguous and file-backed.
3. Pooling widens every distribution. The seven sites differ enough in mean
   irradiance and humidity that shared statistics compress each site's
   own variation toward the pooled mean.

The cost is that the zero-shot transfer study needs statistics for a
site the model never trained on. That is defensible -- a new plant has
an irradiance history long before it has a forecasting model -- but it
is an assumption, and :func:`fit_pooled` exists so the pooled variant
can be reported alongside it rather than argued about.

Why fitted on train years only
------------------------------
A standardiser fitted on all five years has seen the 2020 test distribution.
The leak is small in magnitude and completely invisible in any results table,
which is exactly what makes it worth being strict about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TRAIN_YEARS

__all__ = ["Standardiser", "fit_standardiser", "fit_pooled"]

#: Columns below this standard deviation are treated as constant and
#: passed through unscaled. Dividing by a near-zero sigma turns a
#: numerically dead column into a source of enormous values, which
#: destabilises a network and would break the 4 d.p. C++ comparison for
#: reasons that have nothing to do with the C++.
MIN_SIGMA = 1e-8


@dataclass
class Standardiser:
    """Frozen z-score parameters for one site.

    Attributes
    ----------
    mean, std : dict
        Per-column statistics. Columns absent from these dicts are
        passed through unchanged, which is what keeps categorical and
        already-bounded columns (monsoon phase, cloud type, sin/cos
        encodings) out of the transform.
    site : str
        Site key the statistics were fitted on.
    fit_years : tuple of int
        Years the statistics were fitted on. Recorded so a results table
        can be traced back to the exact fit, and so an accidental
        all-years fit is visible in the artefact rather than only in the
        code that made it.
    """

    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    site: str = ""
    fit_years: tuple[int, ...] = ()
    #: Columns deliberately excluded from scaling.
    passthrough: tuple[str, ...] = ()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise the columns this instance was fitted on.

        Columns present in ``df`` but absent from the fit are left
        untouched rather than raising, so a feature-set variant can be
        transformed by a standardiser fitted on the superset.
        """
        out = df.copy()
        for column in out.columns:
            if column not in self.mean:
                continue
            sigma = self.std.get(column, 1.0)
            sigma = sigma if sigma > MIN_SIGMA else 1.0
            out[column] = (out[column] - self.mean[column]) / sigma
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undo :meth:`transform`."""
        out = df.copy()
        for column in out.columns:
            if column not in self.mean:
                continue
            sigma = self.std.get(column, 1.0)
            sigma = sigma if sigma > MIN_SIGMA else 1.0
            out[column] = out[column] * sigma + self.mean[column]
        return out

    def to_json(self, path: str | Path) -> Path:
        """Serialise to the format SolarInfer's ``FeaturePreprocessor`` loads.

        Sorted keys and an explicit column order so the file is
        byte-stable across runs -- it is hashed into the split manifest
        for the DOI deposit.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "site": self.site,
            "fit_years": list(self.fit_years),
            "columns": sorted(self.mean),
            "mean": {k: float(v) for k, v in sorted(self.mean.items())},
            "std": {k: float(v) for k, v in sorted(self.std.items())},
            "passthrough": list(self.passthrough),
            "min_sigma": MIN_SIGMA,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "Standardiser":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=raw["mean"],
            std=raw["std"],
            site=raw.get("site", ""),
            fit_years=tuple(raw.get("fit_years", ())),
            passthrough=tuple(raw.get("passthrough", ())),
        )


#: Never standardised.
#:
#: Cyclical encodings are already on [-1, 1] and z-scoring them destroys
#: the property that makes them work -- the unit circle stops being a
#: circle. Categorical codes carry no distance, so a mean and a standard
#: deviation are meaningless. Boolean flags are already 0/1.
DEFAULT_PASSTHROUGH: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "monsoon_phase",
    "cloud_type",
    "fill_flag",
    "is_transition",
    "split",
)


def _scalable_columns(df: pd.DataFrame, passthrough) -> list[str]:
    """Numeric columns eligible for scaling, in frame order."""
    banned = set(passthrough)
    out = []
    for column in df.columns:
        base = str(column)
        if base in banned or (base.startswith("kf_") and base[3:] in banned):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if pd.api.types.is_bool_dtype(df[column]):
            continue
        out.append(column)
    return out


def fit_standardiser(
    df: pd.DataFrame,
    site: str,
    *,
    years=TRAIN_YEARS,
    passthrough=DEFAULT_PASSTHROUGH,
) -> Standardiser:
    """Fit z-score parameters on the training years of one site.

    Parameters
    ----------
    df : DataFrame
        UTC-indexed feature frame spanning at least the training years.
    site : str
        Site key, recorded in the artefact.
    years : tuple of int
        Years to fit on. Defaults to ``config.TRAIN_YEARS``; overriding
        this is how the few-shot study fits on a fine-tuning window.

    Notes
    -----
    Statistics are computed with ``ddof=0`` and skip NaN, which is what
    ``numpy`` and every C++ reimplementation will do by default. Using pandas'
    ``ddof=1`` here would put a factor of ``sqrt(n/(n-1))`` between Python and
    C++ that is far too small to see and far too large to pass a 4 d.p. gate.
    """
    mask = np.isin(
        np.asarray(df.index.year, dtype=int), np.asarray(years, dtype=int)
    )
    if not mask.any():
        raise ValueError(f"{site}: no rows in fit years {tuple(years)}")

    train = df.loc[mask]
    columns = _scalable_columns(train, passthrough)

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for column in columns:
        values = train[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        mean[column] = float(finite.mean())
        std[column] = float(finite.std(ddof=0))

    return Standardiser(
        mean=mean,
        std=std,
        site=site,
        fit_years=tuple(int(y) for y in years),
        passthrough=tuple(passthrough),
    )


def fit_pooled(
    frames: dict[str, pd.DataFrame],
    *,
    years=TRAIN_YEARS,
    passthrough=DEFAULT_PASSTHROUGH,
) -> Standardiser:
    """Fit one standardiser across several sites, for the transfer study.

    Zero-shot transfer to an unseen site is the one setting where a
    per-site statistic begs the question: the model is supposed to work
    somewhere it has no history. Pooled statistics answer that honestly,
    at the cost of compressing each site's own variation. Both are
    reported.

    Statistics are pooled over the concatenated training rows, so a site
    with a longer record carries proportionally more weight -- which is
    correct here, since all seven sites have identical coverage.
    """
    train_parts = []
    for site, df in frames.items():
        mask = np.isin(
            np.asarray(df.index.year, dtype=int), np.asarray(years, dtype=int)
        )
        if not mask.any():
            raise ValueError(f"{site}: no rows in fit years {tuple(years)}")
        train_parts.append(df.loc[mask])

    pooled = pd.concat(train_parts, axis=0)
    return fit_standardiser(
        pooled, site="__pooled__", years=years, passthrough=passthrough
    )
