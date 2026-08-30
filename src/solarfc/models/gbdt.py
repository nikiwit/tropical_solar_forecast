"""Gradient-boosted trees, direct multi-horizon.

Strategy
--------
One model per horizon, as the plan specifies. The alternative -- a single model
taking horizon as a feature -- shares statistical strength across horizons but
forces one set of hyperparameters and one feature importance ranking onto
problems that are genuinely different: a 20-minute forecast is dominated by
irradiance persistence, a 36-hour forecast almost entirely by the NWP track.
Keeping them separate is also what makes the per-horizon SHAP analysis
interpretable.

No standardisation
------------------
Trees split on order statistics, so a monotone rescaling of a feature cannot
change the tree they build. :mod:`solarfc.scaling` is therefore not applied
here. It is not unused -- the recurrent and Transformer families need it, and
its serialised form is what SolarInfer's ``FeaturePreprocessor``
loads for the 4 d.p. equivalence check. Applying it to trees would only add a
step for the C++ port to reproduce for no numerical effect.

Early stopping
--------------
Stopped on the validation split (2019), never on the test split. The number of
rounds is a hyperparameter like any other, and choosing it on the test year is
the most common way a benchmark quietly leaks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..dataset import SupervisedSet, to_ghi

__all__ = [
    "ALGORITHMS",
    "GBDTConfig",
    "CATEGORICAL_COLUMNS",
    "fit_gbdt",
    "predict_ghi",
]

ALGORITHMS: tuple[str, ...] = ("xgboost", "lightgbm")

#: Columns that are codes rather than quantities.
#:
#: NSRDB's cloud type is a nominal classification (clear, water, ice,
#: cirrus, overlapping, ...). Left as an integer a tree would split it
#: as if type 7 sat between 6 and 8, which is meaningless. Both
#: libraries support native categorical handling, which partitions the
#: categories instead.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("cloud_type",)


@dataclass
class GBDTConfig:
    """Hyperparameters, recorded with every result row.

    The defaults are deliberately middle-of-the-road rather than tuned.
    This grid exists to establish whether a tree model can beat smart
    persistence at all and to settle the target representation. Optuna
    tuning comes once the target is frozen; tuning before that would
    only mean re-tuning after.
    """

    algorithm: str = "xgboost"
    n_estimators: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    n_jobs: int = -1
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark nominal codes as categorical, leaving everything else untouched."""
    out = frame
    present = [c for c in CATEGORICAL_COLUMNS if c in frame.columns]
    if present:
        out = frame.copy()
        for column in present:
            out[column] = out[column].astype("int32").astype("category")
    return out


def fit_gbdt(
    train: SupervisedSet,
    validation: SupervisedSet,
    config: GBDTConfig | None = None,
):
    """Fit one model for one horizon on the training split.

    Returns the fitted estimator. ``best_iteration`` is available on it
    and is worth recording -- if a model consistently stops at the round
    cap rather than on the validation curve, the cap is the binding
    constraint and the result understates what the family can do.
    """
    config = GBDTConfig() if config is None else config
    if config.algorithm not in ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of {ALGORITHMS}, got {config.algorithm!r}"
        )
    if len(train) == 0 or len(validation) == 0:
        raise ValueError(
            f"empty split: train={len(train)}, validation={len(validation)}"
        )

    X_train, X_val = _prepare(train.X), _prepare(validation.X)

    if config.algorithm == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            reg_lambda=config.reg_lambda,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=config.early_stopping_rounds,
            **config.extra,
        )
        model.fit(
            X_train,
            train.y.to_numpy(),
            eval_set=[(X_val, validation.y.to_numpy())],
            verbose=False,
        )
        return model

    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=-1,
        **config.extra,
    )
    model.fit(
        X_train,
        train.y.to_numpy(),
        eval_set=[(X_val, validation.y.to_numpy())],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False)
        ],
    )
    return model


def predict_ghi(model, data: SupervisedSet) -> np.ndarray:
    """Predict and convert to W/m^2, whatever representation was trained on.

    Every metric in the project is computed on GHI, so the conversion
    belongs here rather than in each caller -- a CSI model scored
    without the rescale would produce numbers between 0 and 2 that look
    like nothing in particular.
    """
    raw = model.predict(_prepare(data.X))
    return to_ghi(raw, data.clearsky_ghi.to_numpy(), data.target)


def best_iteration(model) -> int:
    """Rounds actually used, across both libraries' differing attributes."""
    for attribute in ("best_iteration", "best_iteration_"):
        value = getattr(model, attribute, None)
        if value is not None:
            return int(value)
    return -1
