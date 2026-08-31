"""Observed-past feature engineering, and the guards that keep it trailing.

Two families of test matter here. The first is directional: a lag must look
backwards, and a rolling window must not include a sample it has not reached
yet. The second is the night-time clear-sky index handling, which had a real
bug -- every CSI rolling window demanded a full window of non-NaN samples, which
night makes impossible, so ``delta_csi`` (Module B's entire input) evaluated to
NaN everywhere and every night-origin row was then dropped. That cost 100% of
rows at the 12 h and 36 h horizons. The regression tests below exist so it
cannot come back silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarfc import features as F
from solarfc.config import DAYTIME_CLEARSKY_FLOOR


def _index(days=4):
    return pd.date_range(
        "2020-03-01", periods=days * 144, freq="10min", tz="UTC"
    )


def _nsrdb(index=None, seed=0):
    """A synthetic NSRDB-shaped frame with a real diurnal cycle."""
    index = _index() if index is None else index
    rng = np.random.default_rng(seed)
    n = len(index)

    minute = index.hour * 60 + index.minute
    daylight = np.clip(np.sin(np.pi * (minute - 360) / 720), 0, None)
    clearsky = 1000.0 * daylight
    ghi = clearsky * rng.uniform(0.4, 1.0, n)

    return pd.DataFrame(
        {
            "GHI": ghi,
            "DNI": ghi * 0.8,
            "DHI": ghi * 0.2,
            "Clearsky GHI": clearsky,
            "Clearsky DNI": clearsky * 0.85,
            "Clearsky DHI": clearsky * 0.15,
            "Temperature": 27.0 + rng.normal(0, 1, n),
            "Dew Point": 23.0 + rng.normal(0, 1, n),
            "Relative Humidity": np.clip(80 + rng.normal(0, 5, n), 0, 100),
            "Pressure": 1010.0 + rng.normal(0, 2, n),
            "Wind Speed": np.abs(rng.normal(2, 0.5, n)),
            "Wind Direction": rng.uniform(0, 360, n),
            "Precipitable Water": 4.0 + rng.normal(0, 0.3, n),
            "Surface Albedo": np.full(n, 0.13),
            "Cloud Type": rng.integers(0, 10, n),
            "Fill Flag": np.zeros(n, dtype=int),
            "Solar Zenith Angle": 90.0 - 60.0 * daylight,
            "Aerosol Optical Depth": 0.2 + rng.normal(0, 0.02, n),
            "Alpha": np.full(n, 1.2),
            "Ozone": np.full(n, 0.26),
            "Asymmetry": np.full(n, 0.66),
        },
        index=index,
    )


def _era5(index):
    n = len(index)
    return pd.DataFrame(
        {
            "era5_temp_c": np.full(n, 28.0),
            "era5_dewpoint_c": np.full(n, 24.0),
            "era5_pressure_hpa": np.full(n, 1010.0),
            "era5_cloud_cover": np.full(n, 0.6),
            "era5_precipitable_water_mm": np.full(n, 45.0),
            "era5_wind_speed": np.full(n, 2.0),
            "era5_wind_direction": np.full(n, 180.0),
            "era5_precip_mm_h": np.full(n, 0.3),
            "era5_ssrd_wm2": np.full(n, 200.0),
            "era5_relative_humidity": np.full(n, 80.0),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def built():
    nsrdb = _nsrdb()
    return F.build_observed_past(
        nsrdb, "kuala_lumpur", era5=_era5(nsrdb.index)
    )


class TestDirectionality:
    """A feature must never be a function of its own future."""

    def test_lag_looks_backwards(self, built):
        lag = 6
        assert built["ghi_lag_6"].to_numpy()[lag:] == pytest.approx(
            built["ghi"].to_numpy()[:-lag], nan_ok=True
        )

    def test_rolling_mean_is_trailing(self, built):
        window = 3
        ghi = built["ghi"].to_numpy()
        expected = np.mean([ghi[100 - k] for k in range(window)])
        assert built["ghi_roll_mean_3"].to_numpy()[100] == pytest.approx(
            expected
        )

    def test_perturbing_the_future_leaves_the_past_untouched(self):
        """The decisive leakage test: change tomorrow, yesterday must not move."""
        nsrdb = _nsrdb()
        era5 = _era5(nsrdb.index)
        before = F.build_observed_past(nsrdb, "kuala_lumpur", era5=era5)

        tampered = nsrdb.copy()
        cut = 300
        tampered.iloc[cut:, tampered.columns.get_loc("GHI")] = 9999.0
        after = F.build_observed_past(tampered, "kuala_lumpur", era5=era5)

        head_before = before.iloc[:cut].to_numpy(dtype=float)
        head_after = after.iloc[:cut].to_numpy(dtype=float)
        assert np.allclose(head_before, head_after, equal_nan=True)

    def test_no_known_future_column_leaks_in(self, built):
        F.assert_no_leakage(built.columns)

    def test_assert_no_leakage_actually_fires(self):
        with pytest.raises(AssertionError):
            F.assert_no_leakage(["ghi", "kf_clearsky_ghi"])


class TestClearSkyIndexAtNight:
    """Regression tests for the bug that emptied Module B's input."""

    def test_delta_csi_is_not_all_nan(self, built):
        """This was 100% NaN. It is the entire Module B signal."""
        assert built["delta_csi"].notna().any()

    def test_delta_csi_nan_is_confined_to_the_warm_up(self, built):
        """Past the 24 h window there must be no gap left, night included."""
        series = built["delta_csi"]
        assert series.loc[series.first_valid_index() :].notna().all()

    def test_csi_rolling_windows_survive_the_night(self, built):
        """Every one of these was identically NaN before the fix."""
        for column in (
            "csi_roll_mean_3",
            "csi_roll_mean_144",
            "csi_roll_std_36",
        ):
            series = built[column]
            assert series.notna().any(), column
            assert (
                series.loc[series.first_valid_index() :].notna().all()
            ), column

    def test_csi_is_carried_across_the_night(self, built):
        """No NaN after the first observation -- night reuses the last value."""
        csi = built["csi"]
        first = csi.first_valid_index()
        assert csi.loc[first:].notna().all()

    def test_carried_value_equals_the_last_observation(self, built):
        """Carrying must repeat the real value, not interpolate toward one."""
        csi = built["csi"].to_numpy()
        age = built["csi_age_steps"].to_numpy()
        stale = np.flatnonzero(np.nan_to_num(age, nan=0) > 0)
        picked = stale[len(stale) // 2]
        source = picked - int(age[picked])
        assert csi[picked] == pytest.approx(csi[source])

    def test_age_is_zero_when_observed(self, built):
        """Daytime samples are real observations, so their age is zero."""
        daytime = (
            built["clearsky_ghi_ineichen"].to_numpy() > DAYTIME_CLEARSKY_FLOOR
        )
        age = built["csi_age_steps"].to_numpy()
        observed = daytime & np.isfinite(age)
        assert np.nanmax(age[observed]) == 0

    def test_age_grows_through_the_night(self, built):
        age = built["csi_age_steps"].to_numpy()
        assert np.nanmax(age) > 20

    def test_rolling_mean_ignores_the_carried_value(self):
        """A 24 h CSI mean must average real samples, not one value repeated.

        Computed on the carried series the mean would be dragged toward the last
        observation before sunset. Computed NaN-skip on the raw index it is the
        mean of the day's real samples, which is what Module B compares against.
        """
        nsrdb = _nsrdb()
        built = F.build_observed_past(nsrdb, "kuala_lumpur")
        # Sunset carries the last daytime value; the 24 h mean at a night step
        # must differ from it, or the window collapsed onto the carry.
        age = built["csi_age_steps"].to_numpy()
        night = np.flatnonzero(np.nan_to_num(age, nan=0) > 30)
        night = night[night > 144]
        assert night.size
        at = night[len(night) // 2]
        assert built["csi_roll_mean_144"].to_numpy()[at] != pytest.approx(
            built["csi"].to_numpy()[at]
        )


class TestFeatureSets:
    def test_deployable_excludes_satellite_retrievals(self, built):
        columns = F.feature_columns(built, "deployable")
        for banned in F.SATELLITE_ONLY_COLUMNS:
            assert banned not in columns

    def test_deployable_excludes_derived_satellite_columns(self, built):
        """Dropping 'aod' must also drop 'aod_lag_6' -- the base carries it."""
        columns = F.feature_columns(built, "deployable")
        assert not any(c.startswith("aod_") for c in columns)
        assert not any(c.startswith("nsrdb_clearsky_ghi_") for c in columns)

    def test_deployable_excludes_nsrdb_clearsky(self, built):
        """NSRDB clear-sky is radiative-transfer output, not a measurement."""
        columns = F.feature_columns(built, "deployable")
        assert "nsrdb_clearsky_ghi" not in columns
        assert "nsrdb_solar_zenith" not in columns

    def test_deployable_keeps_the_computable_clearsky(self, built):
        """A plant computes Ineichen from its own coordinates, so this stays."""
        columns = F.feature_columns(built, "deployable")
        assert "clearsky_ghi_ineichen" in columns
        assert "csi" in columns

    def test_fill_flag_is_dropped_from_both_sets(self, built):
        """Retrieval quality is metadata about the label, not an input."""
        for feature_set in F.FEATURE_SETS:
            assert "fill_flag" not in F.feature_columns(built, feature_set)

    def test_full_is_a_strict_superset(self, built):
        full = set(F.feature_columns(built, "full"))
        deployable = set(F.feature_columns(built, "deployable"))
        assert deployable < full

    def test_unknown_feature_set_raises(self, built):
        with pytest.raises(ValueError, match="feature_set"):
            F.feature_columns(built, "everything")


class TestWindDecomposition:
    def test_components_recover_the_speed(self, built):
        speed = np.hypot(
            built["wind_u"].to_numpy(), built["wind_v"].to_numpy()
        )
        assert speed == pytest.approx(built["wind_speed"].to_numpy())

    def test_northerly_wind_has_negative_v(self, built):
        """Meteorological convention: 0 degrees means blowing FROM the north."""
        nsrdb = _nsrdb()
        nsrdb["Wind Direction"] = 0.0
        nsrdb["Wind Speed"] = 5.0
        out = F.build_observed_past(nsrdb, "kuala_lumpur")
        assert out["wind_v"].to_numpy() == pytest.approx(-5.0)
        assert out["wind_u"].to_numpy() == pytest.approx(0.0, abs=1e-12)


class TestStructure:
    def test_era5_is_optional(self):
        nsrdb = _nsrdb()
        out = F.build_observed_past(nsrdb, "kuala_lumpur")
        assert not any(c.startswith("era5_") for c in out.columns)

    def test_index_is_preserved(self, built):
        assert len(built) == len(_index())
        assert built.index.tz is not None

    def test_site_recorded_in_attrs(self, built):
        assert built.attrs["site"] == "kuala_lumpur"

    def test_rolling_min_periods_full_for_defined_series(self):
        """GHI is never NaN, so its windows demand the whole window."""
        assert F._min_periods(144, carried=False) == 144

    def test_rolling_min_periods_fractional_for_carried_series(self):
        assert F._min_periods(144, carried=True) == 36


class TestStepsSinceObserved:
    def test_counts_from_the_last_finite_sample(self):
        values = np.array([1.0, np.nan, np.nan, 2.0, np.nan])
        assert F._steps_since_observed(values).tolist() == [
            0.0,
            1.0,
            2.0,
            0.0,
            1.0,
        ]

    def test_leading_gap_is_nan(self):
        out = F._steps_since_observed(np.array([np.nan, np.nan, 1.0]))
        assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 0.0
