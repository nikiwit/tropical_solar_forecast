"""Chronological splits, monsoon labelling and transition windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarfc import splits as S
from solarfc.config import (
    NE_MONSOON,
    SW_MONSOON,
    INTER_MONSOON_I,
    INTER_MONSOON_II,
)


def _index(start="2016-01-01", periods=1000, freq="10min"):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


class TestMonsoonPhase:
    @pytest.mark.parametrize(
        "month,expected",
        [
            (1, NE_MONSOON),
            (3, NE_MONSOON),
            (4, INTER_MONSOON_I),
            (7, SW_MONSOON),
            (9, SW_MONSOON),
            (10, INTER_MONSOON_II),
            (12, NE_MONSOON),
        ],
    )
    def test_month_mapping(self, month, expected):
        idx = pd.DatetimeIndex([pd.Timestamp(2018, month, 15, tz="UTC")])
        assert S.monsoon_phase(idx)[0] == expected

    def test_every_month_is_labelled(self):
        idx = pd.date_range("2018-01-01", "2018-12-31", freq="D", tz="UTC")
        phases = S.monsoon_phase(idx)
        assert set(np.unique(phases)) == {0, 1, 2, 3}


class TestTransitionWindows:
    def test_transition_date_is_inside_window(self):
        idx = pd.DatetimeIndex([pd.Timestamp(2018, 5, 1, tz="UTC")])
        assert S.is_transition_window(idx)[0]

    def test_mid_season_is_outside_window(self):
        idx = pd.DatetimeIndex([pd.Timestamp(2018, 7, 15, tz="UTC")])
        assert not S.is_transition_window(idx)[0]

    def test_window_half_width_respected(self):
        inside = pd.DatetimeIndex([pd.Timestamp(2018, 5, 8, tz="UTC")])
        outside = pd.DatetimeIndex([pd.Timestamp(2018, 6, 10, tz="UTC")])
        assert S.is_transition_window(inside, window_days=10)[0]
        assert not S.is_transition_window(outside, window_days=10)[0]

    def test_windows_stay_a_minority_stratum(self):
        """At +/-21 days the April and May windows merged and covered ~40% of
        the year, destroying the contrast the stratum exists to provide."""
        idx = pd.date_range("2018-01-01", "2018-12-31", freq="D", tz="UTC")
        coverage = S.is_transition_window(idx).mean()
        assert 0.15 < coverage < 0.30

    def test_leap_year_handled(self):
        """Day-of-year anchors must shift in a leap year, not drift by a day."""
        idx = pd.DatetimeIndex([pd.Timestamp(2020, 5, 1, tz="UTC")])
        assert S.is_transition_window(idx)[0]

    def test_zero_window_marks_only_exact_dates(self):
        on = pd.DatetimeIndex([pd.Timestamp(2018, 5, 1, tz="UTC")])
        off = pd.DatetimeIndex([pd.Timestamp(2018, 5, 2, tz="UTC")])
        assert S.is_transition_window(on, window_days=0)[0]
        assert not S.is_transition_window(off, window_days=0)[0]

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="window_days"):
            S.is_transition_window(_index(), window_days=-1)


class TestSplitAssignment:
    def _frame(self):
        idx = pd.date_range("2016-01-01", "2020-12-31", freq="D", tz="UTC")
        return pd.DataFrame(
            {"GHI": np.arange(len(idx), dtype=float)}, index=idx
        )

    def test_years_route_to_expected_splits(self):
        out = S.assign_splits(self._frame())
        assert set(out.loc["2016":"2018", "split"]) == {"train"}
        assert set(out.loc["2019", "split"]) == {"val"}
        assert set(out.loc["2020", "split"]) == {"test"}

    def test_splits_do_not_overlap_in_time(self):
        """The property that makes the benchmark honest."""
        out = S.assign_splits(self._frame())
        train_end = out[out["split"] == "train"].index.max()
        val_start = out[out["split"] == "val"].index.min()
        val_end = out[out["split"] == "val"].index.max()
        test_start = out[out["split"] == "test"].index.min()
        assert train_end < val_start
        assert val_end < test_start

    def test_input_is_not_mutated(self):
        df = self._frame()
        S.assign_splits(df)
        assert list(df.columns) == ["GHI"]

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"GHI": [1.0, 2.0]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            S.assign_splits(df)

    def test_manifest_is_written_and_hashed(self, tmp_path):
        out = S.assign_splits(self._frame())
        path = tmp_path / "splits.json"
        manifest = S.write_split_manifest(out, path)
        assert path.exists()
        assert len(manifest["sha256"]) == 64
        assert {s["split"] for s in manifest["splits"]} == {
            "train",
            "val",
            "test",
        }

    def test_manifest_hash_is_deterministic(self, tmp_path):
        out = S.assign_splits(self._frame())
        a = S.write_split_manifest(out, tmp_path / "a.json")
        b = S.write_split_manifest(out, tmp_path / "b.json")
        assert a["sha256"] == b["sha256"]
