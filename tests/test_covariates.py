"""Known-future covariates, and the leakage guards on them.

The leakage tests matter most here. Known-future covariates are the one place
in the pipeline where future information is legitimate, which makes them the
one place where genuine leakage would be hardest to spot by reading the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarfc import covariates as C


def _index(start="2020-06-15 00:00", periods=144, freq="10min"):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


def _era5(index):
    return pd.DataFrame(
        {
            "era5_cloud_cover": np.full(len(index), 0.5),
            "era5_temp_c": np.full(len(index), 28.0),
            "era5_dewpoint_c": np.full(len(index), 24.0),
            "era5_relative_humidity": np.full(len(index), 80.0),
            "era5_precip_mm_h": np.full(len(index), 0.5),
        },
        index=index,
    )


class TestTrackContents:
    def test_nwp_free_has_no_atmospheric_column(self):
        """The defining property of the lower-bound track."""
        cols = C.known_future_columns("nwp_free")
        assert not any(c.startswith("era5_") for c in cols)

    def test_nwp_tracks_carry_atmosphere(self):
        for track in ("realistic", "perfect"):
            cols = C.known_future_columns(track)
            assert all(f in cols for f in C.NWP_FEATURES)

    def test_no_observed_irradiance_in_any_track(self):
        """GHI is an observation, not a forecast. It must never reach the decoder."""
        for track in C.TRACKS:
            cols = C.known_future_columns(track)
            assert "GHI" not in cols
            assert "csi" not in cols
            assert not any("ghi" == c.lower() for c in cols)

    def test_clearsky_ghi_is_allowed(self):
        """Clear-sky GHI is computed from geometry, not measured, so it is fine."""
        assert "clearsky_ghi" in C.known_future_columns("nwp_free")

    def test_invalid_track_raises(self):
        with pytest.raises(ValueError, match="track must be one of"):
            C.known_future_columns("optimistic")


class TestCalendarFeatures:
    def test_cyclical_encoding_wraps(self):
        """23:50 and 00:00 must be adjacent, not maximally distant."""
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01 23:50", tz="UTC"),
             pd.Timestamp("2020-01-02 00:00", tz="UTC")]
        )
        out = C.calendar_features(idx)
        gap = np.hypot(
            out["hour_sin"].iloc[1] - out["hour_sin"].iloc[0],
            out["hour_cos"].iloc[1] - out["hour_cos"].iloc[0],
        )
        assert gap < 0.05

    def test_midnight_and_noon_are_opposite(self):
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01 00:00", tz="UTC"),
             pd.Timestamp("2020-01-01 12:00", tz="UTC")]
        )
        out = C.calendar_features(idx)
        assert out["hour_cos"].iloc[0] == pytest.approx(1.0)
        assert out["hour_cos"].iloc[1] == pytest.approx(-1.0)

    def test_monsoon_phase_labelled(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2020-07-15", tz="UTC")])
        assert C.calendar_features(idx)["monsoon_phase"].iloc[0] == 2  # SW


class TestSolarGeometry:
    def test_zenith_below_90_at_local_noon(self):
        # KL is UTC+8, so solar noon is around 05:00 UTC.
        idx = pd.DatetimeIndex([pd.Timestamp("2020-06-15 05:00", tz="UTC")])
        out = C.solar_geometry(idx, "kuala_lumpur")
        assert out["solar_zenith"].iloc[0] < 30.0

    def test_zenith_above_90_at_local_midnight(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2020-06-15 17:00", tz="UTC")])
        out = C.solar_geometry(idx, "kuala_lumpur")
        assert out["solar_zenith"].iloc[0] > 90.0

    def test_clearsky_zero_at_night_positive_by_day(self):
        night = C.solar_geometry(
            pd.DatetimeIndex([pd.Timestamp("2020-06-15 17:00", tz="UTC")]),
            "kuala_lumpur",
        )
        day = C.solar_geometry(
            pd.DatetimeIndex([pd.Timestamp("2020-06-15 05:00", tz="UTC")]),
            "kuala_lumpur",
        )
        assert night["clearsky_ghi"].iloc[0] == pytest.approx(0.0, abs=1.0)
        assert day["clearsky_ghi"].iloc[0] > 700.0

    def test_cos_zenith_negative_at_night(self):
        out = C.solar_geometry(
            pd.DatetimeIndex([pd.Timestamp("2020-06-15 17:00", tz="UTC")]),
            "kuala_lumpur",
        )
        assert out["cos_zenith"].iloc[0] < 0.0

    def test_southern_site_differs_from_northern(self):
        """Jakarta is south of the equator; geometry must reflect that."""
        idx = pd.DatetimeIndex([pd.Timestamp("2020-12-21 05:00", tz="UTC")])
        kl = C.solar_geometry(idx, "kuala_lumpur")["solar_zenith"].iloc[0]
        jk = C.solar_geometry(idx, "jakarta")["solar_zenith"].iloc[0]
        assert abs(kl - jk) > 3.0


class TestBuildKnownFuture:
    def test_nwp_free_needs_no_era5(self):
        idx = _index()
        out = C.build_known_future(idx, "kuala_lumpur", track="nwp_free")
        assert list(out.columns) == list(C.known_future_columns("nwp_free"))
        assert len(out) == len(idx)

    def test_perfect_requires_era5(self):
        with pytest.raises(ValueError, match="requires ERA5"):
            C.build_known_future(_index(), "kuala_lumpur", track="perfect")

    def test_realistic_requires_lead_hours(self):
        """Without lead times the degradation would silently apply one
        magnitude to every horizon."""
        idx = _index()
        with pytest.raises(ValueError, match="requires lead_hours"):
            C.build_known_future(
                idx, "kuala_lumpur", track="realistic", era5=_era5(idx)
            )

    def test_missing_era5_columns_raise(self):
        idx = _index()
        bad = _era5(idx).drop(columns=["era5_cloud_cover"])
        with pytest.raises(ValueError, match="missing required columns"):
            C.build_known_future(idx, "kuala_lumpur", track="perfect", era5=bad)

    def test_perfect_passes_era5_through_unmodified(self):
        idx = _index()
        era5 = _era5(idx)
        out = C.build_known_future(idx, "kuala_lumpur", track="perfect", era5=era5)
        np.testing.assert_allclose(
            out["era5_cloud_cover"].to_numpy(), era5["era5_cloud_cover"].to_numpy()
        )

    def test_realistic_actually_differs_from_perfect(self):
        idx = _index()
        era5 = _era5(idx)
        lead = C.lead_hours_for_window(idx[0], idx)

        perfect = C.build_known_future(
            idx, "kuala_lumpur", track="perfect", era5=era5
        )
        realistic = C.build_known_future(
            idx, "kuala_lumpur", track="realistic", era5=era5,
            lead_hours=lead, rng=np.random.default_rng(0),
        )
        assert not np.allclose(
            perfect["era5_cloud_cover"].to_numpy(),
            realistic["era5_cloud_cover"].to_numpy(),
        )

    def test_column_order_stable_across_calls(self):
        idx = _index()
        era5 = _era5(idx)
        a = C.build_known_future(idx, "kuala_lumpur", track="perfect", era5=era5)
        b = C.build_known_future(idx, "kuala_lumpur", track="perfect", era5=era5)
        assert list(a.columns) == list(b.columns)


class TestDegradation:
    def test_error_grows_with_lead_time(self):
        model = C.PROVISIONAL_ERROR_MODELS["era5_cloud_cover"]
        assert model.sigma_at(1.0) < model.sigma_at(24.0) < model.sigma_at(48.0)

    def test_error_saturates(self):
        """Forecast error saturates at climatological spread, not unbounded."""
        model = C.PROVISIONAL_ERROR_MODELS["era5_cloud_cover"]
        assert model.sigma_at(10_000.0) == pytest.approx(model.sigma_max)

    def test_cloud_cover_stays_in_physical_bounds(self):
        model = C.PROVISIONAL_ERROR_MODELS["era5_cloud_cover"]
        rng = np.random.default_rng(1)
        out = C.degrade(np.full(5000, 0.95), model, np.full(5000, 48.0), rng=rng)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_precipitation_never_negative(self):
        model = C.PROVISIONAL_ERROR_MODELS["era5_precip_mm_h"]
        rng = np.random.default_rng(2)
        out = C.degrade(np.zeros(5000), model, np.full(5000, 48.0), rng=rng)
        assert out.min() >= 0.0

    def test_noise_is_temporally_correlated_not_white(self):
        """Real forecast errors persist. White noise would let a model average
        the error away across the window and understate the damage."""
        rng = np.random.default_rng(3)
        correlated = C._correlated_noise(20_000, correlation_steps=36.0, rng=rng)
        white = C._correlated_noise(20_000, correlation_steps=1.0, rng=rng)

        lag1_corr = np.corrcoef(correlated[:-1], correlated[1:])[0, 1]
        lag1_white = np.corrcoef(white[:-1], white[1:])[0, 1]
        assert lag1_corr > 0.9
        assert abs(lag1_white) < 0.05

    def test_correlated_noise_keeps_unit_variance(self):
        """Otherwise a longer correlation length would silently shrink the error."""
        rng = np.random.default_rng(4)
        for steps in (1.0, 6.0, 36.0):
            noise = C._correlated_noise(50_000, steps, rng)
            assert noise.std() == pytest.approx(1.0, abs=0.05)

    def test_degradation_is_reproducible_with_a_seed(self):
        model = C.PROVISIONAL_ERROR_MODELS["era5_temp_c"]
        values = np.full(100, 28.0)
        lead = np.full(100, 24.0)
        a = C.degrade(values, model, lead, rng=np.random.default_rng(7))
        b = C.degrade(values, model, lead, rng=np.random.default_rng(7))
        np.testing.assert_allclose(a, b)

    def test_longer_leads_produce_larger_spread(self):
        model = C.PROVISIONAL_ERROR_MODELS["era5_temp_c"]
        rng = np.random.default_rng(5)
        near = C.degrade(np.full(20_000, 28.0), model, np.full(20_000, 1.0), rng=rng)
        far = C.degrade(np.full(20_000, 28.0), model, np.full(20_000, 48.0), rng=rng)
        assert far.std() > near.std()


class TestLeadHours:
    def test_zero_at_origin(self):
        idx = _index(periods=10)
        lead = C.lead_hours_for_window(idx[0], idx)
        assert lead[0] == pytest.approx(0.0)

    def test_matches_step_size(self):
        idx = _index(periods=10)
        lead = C.lead_hours_for_window(idx[0], idx)
        assert lead[6] == pytest.approx(1.0)  # 6 steps x 10 min

    def test_negative_before_origin_not_clipped(self):
        """A caller mistake should surface, not be silently absorbed."""
        idx = _index(periods=10)
        lead = C.lead_hours_for_window(idx[5], idx)
        assert lead[0] < 0.0


class TestDecoderWindow:
    def test_covers_the_longest_horizon(self):
        from solarfc.config import HORIZON_STEPS

        assert C.MAX_DECODER_STEPS == max(HORIZON_STEPS)

    def test_reaches_the_day_ahead_submission_window(self):
        """Day-ahead is issued at 10:00 and must cover to end of next day: 38 h."""
        from solarfc.config import STEP_MINUTES

        assert C.MAX_DECODER_STEPS * STEP_MINUTES / 60.0 >= 38.0
