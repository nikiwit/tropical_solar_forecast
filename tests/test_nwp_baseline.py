"""The operational NWP baseline, and the guards that keep it a baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarfc import nwp_baseline as N


def _forecast_frame(n: int = 2000, lead: int = 24, seed: int = 0):
    """Synthetic hourly forecast fields with a known cloud-CSI relation."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=n, freq="1h", tz="UTC")
    cloud = rng.uniform(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            f"era5_cloud_cover_lead{lead}h": cloud,
            f"era5_temp_c_lead{lead}h": 27.0 + rng.normal(0, 2, n),
            f"era5_relative_humidity_lead{lead}h": 80.0 + rng.normal(0, 5, n),
            f"era5_dewpoint_c_lead{lead}h": 23.0 + rng.normal(0, 2, n),
            f"era5_precip_mm_h_lead{lead}h": rng.gamma(0.4, 0.8, n),
        },
        index=index,
    )
    return frame, cloud


class TestTransmittance:
    def test_bounded_and_monotone(self):
        cc = np.linspace(0.0, 1.0, 51)
        tau = N.transmittance(cc, *N.KASTEN_CZEPLAK)
        assert np.all(tau <= 1.0)
        assert np.all(tau >= N.TRANSMITTANCE_FLOOR)
        # More cloud can never mean more irradiance.
        assert np.all(np.diff(tau) <= 1e-12)

    def test_clear_sky_is_unattenuated(self):
        assert N.transmittance(0.0, 0.75, 3.4) == pytest.approx(1.0)

    def test_published_coefficients_at_overcast(self):
        """The value that does not transfer to the tropics -- see the module."""
        assert N.transmittance(1.0, *N.KASTEN_CZEPLAK) == pytest.approx(0.25)

    def test_out_of_range_cloud_is_clipped_not_raised(self):
        tau = N.transmittance([-0.2, 1.4], 0.5, 2.0)
        assert tau[0] == pytest.approx(1.0)
        assert tau[1] == pytest.approx(0.5)

    def test_nan_propagates(self):
        assert np.isnan(N.transmittance([np.nan], 0.5, 2.0)[0])


class TestFitTransmittance:
    def test_recovers_known_coefficients(self):
        cc = np.linspace(0.0, 1.0, 4000)
        truth = N.transmittance(cc, 0.40, 0.80)
        a, b = N.fit_transmittance(cc, truth)
        assert a == pytest.approx(0.40, abs=0.02)
        assert b == pytest.approx(0.80, abs=0.05)

    def test_falls_back_to_published_on_thin_data(self):
        a, b = N.fit_transmittance([0.5] * 10, [0.6] * 10)
        assert (a, b) == N.KASTEN_CZEPLAK

    def test_ignores_non_finite_pairs(self):
        cc = np.linspace(0.0, 1.0, 4000)
        y = N.transmittance(cc, 0.40, 0.80)
        y[::7] = np.nan
        a, b = N.fit_transmittance(cc, y)
        assert a == pytest.approx(0.40, abs=0.02)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            N.fit_transmittance(np.zeros(10), np.zeros(11))


class TestCloudDrivenGhi:
    def test_night_stays_zero_whatever_the_cloud(self):
        clearsky = np.array([0.0, 0.0, 500.0])
        out = N.cloud_driven_ghi([0.0, 1.0, 0.0], clearsky, 0.4, 0.8)
        assert out[0] == 0.0 and out[1] == 0.0
        assert out[2] == pytest.approx(500.0)

    def test_never_exceeds_the_clear_sky_envelope(self):
        rng = np.random.default_rng(1)
        clearsky = rng.uniform(0, 1000, 500)
        out = N.cloud_driven_ghi(rng.uniform(0, 1, 500), clearsky, 0.4, 0.8)
        assert np.all(out <= clearsky + 1e-9)


class TestLeadForHorizon:
    @pytest.mark.parametrize(
        "steps,expected",
        [
            (2, 24),  # 20 min
            (36, 24),  # 6 h
            (144, 24),  # 24 h -- the mapping is exact here
            (216, 48),  # 36 h
            (288, 48),  # 48 h
        ],
    )
    def test_shortest_admissible_lead(self, steps, expected):
        assert N.lead_for_horizon(steps) == expected

    def test_analysis_is_never_returned(self):
        """Lead 0 is a reanalysis-time value, not a forecast."""
        assert N.lead_for_horizon(1) != 0

    def test_beyond_the_archive_falls_back_to_the_longest_lead(self):
        assert N.lead_for_horizon(10_000) == max(N.LEAD_HOURS)


class TestMos:
    def test_fit_predict_recovers_a_linear_signal(self):
        frame, cloud = _forecast_frame()
        csi = 0.9 - 0.3 * cloud
        model = N.fit_mos(frame, csi, lead_hours=24)
        predicted = model.predict_csi(frame)
        assert np.corrcoef(predicted, csi)[0, 1] > 0.95

    def test_predictions_are_clipped_to_the_csi_range(self):
        frame, cloud = _forecast_frame()
        model = N.fit_mos(frame, 0.9 - 0.3 * cloud, lead_hours=24)
        predicted = model.predict_csi(frame)
        assert np.nanmin(predicted) >= 0.0

    def test_missing_forecast_field_yields_nan_not_a_guess(self):
        frame, cloud = _forecast_frame()
        model = N.fit_mos(frame, 0.9 - 0.3 * cloud, lead_hours=24)
        broken = frame.copy()
        broken.iloc[5, 0] = np.nan
        assert np.isnan(model.predict_csi(broken)[5])

    def test_ghi_is_csi_times_clearsky(self):
        frame, cloud = _forecast_frame()
        model = N.fit_mos(frame, 0.9 - 0.3 * cloud, lead_hours=24)
        clearsky = np.full(len(frame), 800.0)
        np.testing.assert_allclose(
            model.predict_ghi(frame, clearsky),
            model.predict_csi(frame) * 800.0,
        )

    def test_refuses_to_fit_on_too_few_rows(self):
        frame, cloud = _forecast_frame(n=50)
        with pytest.raises(ValueError, match="usable rows"):
            N.fit_mos(frame, 0.9 - 0.3 * cloud, lead_hours=24)

    def test_wrong_lead_has_no_design_matrix(self):
        frame, cloud = _forecast_frame(lead=24)
        with pytest.raises(ValueError, match="no forecast fields"):
            N.fit_mos(frame, 0.9 - 0.3 * cloud, lead_hours=48)

    def test_fitting_cannot_see_the_evaluation_period(self):
        """Rows withheld from the fit must not move its coefficients."""
        frame, cloud = _forecast_frame(n=3000)
        csi = 0.9 - 0.3 * cloud
        train = np.arange(len(frame)) < 1500

        before = N.fit_mos(frame[train], csi[train], lead_hours=24)
        tampered = frame.copy()
        tampered.iloc[1500:, 0] = 0.0
        after = N.fit_mos(tampered[train], csi[train], lead_hours=24)

        np.testing.assert_allclose(before.coef, after.coef)
        assert before.intercept == pytest.approx(after.intercept)


class TestUpsampleForecast:
    def test_hourly_values_survive_on_the_hour(self):
        frame, _ = _forecast_frame(n=48)
        target = pd.date_range(
            frame.index[0], frame.index[-1], freq="10min", tz="UTC"
        )
        out = N.upsample_forecast(frame, target)
        on_hour = out.loc[frame.index]
        np.testing.assert_allclose(
            on_hour.to_numpy(), frame.to_numpy(), atol=1e-9
        )

    def test_interpolates_between_samples(self):
        index = pd.date_range("2018-01-01", periods=2, freq="1h", tz="UTC")
        frame = pd.DataFrame({"era5_cloud_cover_lead24h": [0.0, 1.0]}, index)
        target = pd.date_range(index[0], index[-1], freq="30min", tz="UTC")
        out = N.upsample_forecast(frame, target)
        assert out.iloc[1, 0] == pytest.approx(0.5)

    def test_does_not_extrapolate_past_the_forecast(self):
        frame, _ = _forecast_frame(n=24)
        target = pd.date_range(
            frame.index[0],
            frame.index[-1] + pd.Timedelta("6h"),
            freq="10min",
            tz="UTC",
        )
        out = N.upsample_forecast(frame, target)
        assert out.iloc[-1].isna().all()
