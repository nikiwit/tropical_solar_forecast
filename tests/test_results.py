"""The results schema every model writes into.

The properties that matter are consistency ones. Every model must be scored on
the same samples, the aggregate must agree with its own breakdown, and a row
must carry enough provenance to be traced back to the code that made it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarfc import results as R


def _index(days=40):
    return pd.date_range(
        "2020-01-01", periods=days * 144, freq="10min", tz="UTC"
    )


def _series(index, seed=0):
    """Truth, a prediction and a clear-sky envelope with a diurnal cycle."""
    rng = np.random.default_rng(seed)
    minute = index.hour * 60 + index.minute
    envelope = 1000.0 * np.clip(np.sin(np.pi * (minute - 360) / 720), 0, None)
    truth = envelope * rng.uniform(0.3, 1.0, len(index))
    prediction = truth + rng.normal(0, 40, len(index))
    return truth, prediction, envelope


class TestScorePredictions:
    def test_returns_one_row_per_stratum(self):
        index = _index()
        truth, prediction, envelope = _series(index)
        out = R.score_predictions(truth, prediction, envelope, index)
        assert len(out) == out["stratum"].nunique()
        assert "all" in set(out["stratum"])

    def test_aggregate_uses_only_daytime(self):
        """Night is excluded, so n must be well under the row count."""
        index = _index()
        truth, prediction, envelope = _series(index)
        out = R.score_predictions(truth, prediction, envelope, index)
        aggregate = out[out["stratum"] == "all"].iloc[0]
        assert aggregate["n"] < len(index)
        assert aggregate["n"] == int((envelope > 20).sum())

    def test_strata_partition_the_aggregate(self):
        """Monsoon strata must sum to the aggregate, or the breakdown and the
        headline number are describing different sample sets."""
        index = _index(days=300)
        truth, prediction, envelope = _series(index)
        out = R.score_predictions(truth, prediction, envelope, index)
        aggregate = out[out["stratum"] == "all"]["n"].iloc[0]
        monsoon = out[out["stratum_kind"] == "monsoon"]["n"].sum()
        assert monsoon == aggregate

    def test_regime_strata_also_partition(self):
        index = _index(days=300)
        truth, prediction, envelope = _series(index)
        out = R.score_predictions(truth, prediction, envelope, index)
        aggregate = out[out["stratum"] == "all"]["n"].iloc[0]
        regime = out[out["stratum_kind"] == "regime"]["n"].sum()
        assert regime == aggregate

    def test_forecast_skill_is_nan_without_a_reference(self):
        index = _index()
        truth, prediction, envelope = _series(index)
        out = R.score_predictions(truth, prediction, envelope, index)
        assert out["fs_smart"].isna().all()

    def test_forecast_skill_positive_against_a_worse_reference(self):
        index = _index()
        truth, prediction, envelope = _series(index)
        worse = truth + np.random.default_rng(1).normal(0, 200, len(index))
        out = R.score_predictions(
            truth, prediction, envelope, index, reference_smart=worse
        )
        assert out[out["stratum"] == "all"]["fs_smart"].iloc[0] > 0

    def test_perfect_prediction_scores_zero_error(self):
        index = _index()
        truth, _, envelope = _series(index)
        out = R.score_predictions(truth, truth, envelope, index)
        aggregate = out[out["stratum"] == "all"].iloc[0]
        assert aggregate["mae"] == pytest.approx(0.0, abs=1e-9)
        assert aggregate["r2"] == pytest.approx(1.0)

    def test_stratify_off_gives_only_the_aggregate(self):
        index = _index()
        truth, prediction, envelope = _series(index)
        out = R.score_predictions(
            truth, prediction, envelope, index, stratify=False
        )
        assert len(out) == 1 and out["stratum"].iloc[0] == "all"

    def test_shape_mismatch_raises(self):
        index = _index(days=2)
        truth, prediction, envelope = _series(index)
        with pytest.raises(ValueError, match="shape mismatch"):
            R.score_predictions(truth[:-1], prediction, envelope, index)


class TestAppendResults:
    def test_writes_the_full_schema(self, tmp_path):
        index = _index()
        truth, prediction, envelope = _series(index)
        scored = R.score_predictions(truth, prediction, envelope, index)
        meta = R.RunMeta(run_id="test-1", model="xgboost")

        path = R.append_results(
            scored,
            tmp_path / "results.csv",
            meta=meta,
            site="kuala_lumpur",
            horizon_label="6h",
            horizon_steps=36,
            track="realistic",
            feature_set="full",
            target="csi",
            split="test",
        )
        frame = pd.read_csv(path)
        assert list(frame.columns) == list(R.RESULT_COLUMNS)
        assert (frame["model"] == "xgboost").all()
        assert (frame["site"] == "kuala_lumpur").all()

    def test_appends_without_duplicating_the_header(self, tmp_path):
        index = _index()
        truth, prediction, envelope = _series(index)
        scored = R.score_predictions(truth, prediction, envelope, index)
        meta = R.RunMeta(run_id="test-2", model="lightgbm")
        target = tmp_path / "results.csv"

        for horizon in ("1h", "6h"):
            R.append_results(scored, target, meta=meta, horizon_label=horizon)

        frame = pd.read_csv(target)
        assert set(frame["horizon_label"]) == {"1h", "6h"}
        assert len(frame) == 2 * len(scored)

    def test_load_missing_file_returns_empty_schema(self, tmp_path):
        frame = R.load_results(tmp_path / "absent.csv")
        assert frame.empty
        assert list(frame.columns) == list(R.RESULT_COLUMNS)


class TestRunMeta:
    def test_records_provenance(self):
        meta = R.RunMeta(run_id="r", model="m")
        assert meta.git_commit
        assert meta.timestamp.startswith("20")
        assert meta.python

    def test_serialises(self, tmp_path):
        import json

        meta = R.RunMeta(
            run_id="r", model="m", hyperparameters={"max_depth": 8}
        )
        path = meta.to_json(tmp_path / "run.json")
        loaded = json.loads(path.read_text())
        assert loaded["hyperparameters"]["max_depth"] == 8


class TestPivotHorizons:
    def test_orders_horizons_chronologically(self, tmp_path):
        """'2h' must not sort between '18h' and '20min'."""
        rows = []
        for horizon in ("20min", "2h", "18h", "6h"):
            rows.append(
                {
                    "model": "xgboost",
                    "track": "realistic",
                    "feature_set": "full",
                    "split": "test",
                    "stratum": "all",
                    "horizon_label": horizon,
                    "mae": 100.0,
                }
            )
        table = R.pivot_horizons(pd.DataFrame(rows))
        assert list(table.columns) == ["20min", "2h", "6h", "18h"]
