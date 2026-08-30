"""Ramp detection and scoring."""

from __future__ import annotations

import numpy as np
import pytest

from solarfc import ramp as R


class TestDetectRamps:
    def test_clear_day_has_no_ramps(self):
        cs = np.full(20, 800.0)
        ghi = cs * 0.95
        assert not R.detect_ramps(ghi, cs).any()

    def test_convective_collapse_is_detected(self):
        cs = np.full(20, 800.0)
        ghi = np.full(20, 760.0)
        ghi[10:] = 150.0  # ~76% drop, far beyond the 50% threshold
        assert R.detect_ramps(ghi, cs)[10]

    def test_night_is_never_a_ramp(self):
        """Clear-sky floor must dominate: a big swing at night is not a ramp."""
        cs = np.full(20, 5.0)
        ghi = np.zeros(20)
        ghi[10:] = 500.0
        assert not R.detect_ramps(ghi, cs).any()

    def test_leading_window_cannot_be_an_event(self):
        cs = np.full(20, 800.0)
        ghi = np.full(20, 100.0)
        assert not R.detect_ramps(ghi, cs, window_steps=3)[:3].any()

    def test_threshold_normalised_by_clearsky_not_absolute(self):
        """Same absolute drop: a ramp at low sun, routine at solar noon."""
        drop = 200.0
        low_cs = np.full(10, 300.0)
        high_cs = np.full(10, 1000.0)

        low = np.full(10, 280.0)
        low[5:] -= drop
        high = np.full(10, 950.0)
        high[5:] -= drop

        assert R.detect_ramps(low, low_cs)[5]
        assert not R.detect_ramps(high, high_cs)[5]

    def test_signed_direction(self):
        cs = np.full(20, 800.0)
        down = np.full(20, 760.0)
        down[10:] = 100.0
        up = np.full(20, 100.0)
        up[10:] = 760.0

        assert R.detect_ramps(down, cs, signed=True)[10] == -1
        assert R.detect_ramps(up, cs, signed=True)[10] == 1

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window_steps"):
            R.detect_ramps(np.zeros(5), np.ones(5), window_steps=0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            R.detect_ramps(np.zeros(5), np.ones(6))


class TestLeadTime:
    def test_exact_hit_has_zero_lead(self):
        obs = np.zeros(20, dtype=bool)
        pred = np.zeros(20, dtype=bool)
        obs[10] = pred[10] = True
        matched, leads = R.ramp_detection_lead_time(obs, pred)
        assert matched.tolist() == [10]
        assert leads.tolist() == [0.0]

    def test_early_prediction_gives_positive_lead(self):
        obs = np.zeros(20, dtype=bool)
        pred = np.zeros(20, dtype=bool)
        obs[10] = True
        pred[8] = True  # two steps early = 20 minutes at 10-min resolution
        _, leads = R.ramp_detection_lead_time(obs, pred, tolerance_steps=3)
        assert leads.tolist() == [20.0]

    def test_late_prediction_gives_negative_lead(self):
        obs = np.zeros(20, dtype=bool)
        pred = np.zeros(20, dtype=bool)
        obs[10] = True
        pred[12] = True
        _, leads = R.ramp_detection_lead_time(obs, pred, tolerance_steps=3)
        assert leads.tolist() == [-20.0]

    def test_outside_tolerance_is_a_miss(self):
        obs = np.zeros(30, dtype=bool)
        pred = np.zeros(30, dtype=bool)
        obs[10] = True
        pred[25] = True
        matched, _ = R.ramp_detection_lead_time(obs, pred, tolerance_steps=3)
        assert matched.size == 0

    def test_one_prediction_cannot_claim_two_events(self):
        """Guards against a spray-predicting model inflating recall."""
        obs = np.zeros(30, dtype=bool)
        obs[[10, 11]] = True
        pred = np.zeros(30, dtype=bool)
        pred[10] = True
        matched, _ = R.ramp_detection_lead_time(obs, pred, tolerance_steps=3)
        assert matched.size == 1


class TestRampMetrics:
    def _series(self):
        cs = np.full(60, 800.0)
        truth = np.full(60, 760.0)
        truth[20:26] = 100.0  # a real collapse and recovery
        return cs, truth

    def test_perfect_forecast_scores_one(self):
        cs, truth = self._series()
        m = R.ramp_metrics(truth, truth, cs)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.n_true_positive == m.n_observed

    def test_flat_forecast_misses_every_ramp(self):
        """The failure mode ramp metrics exist to expose."""
        cs, truth = self._series()
        flat = np.full(60, float(np.mean(truth)))
        m = R.ramp_metrics(truth, flat, cs)
        assert m.n_observed > 0
        assert m.n_predicted == 0
        assert m.recall == pytest.approx(0.0)
        assert np.isnan(m.precision)  # undefined, not zero, with no predictions

    def test_smooth_forecast_can_have_low_rmse_yet_miss_ramps(self):
        """RMSE alone would rate this model acceptably; recall reveals it."""
        from solarfc.metrics import rmse

        cs, truth = self._series()
        # Heavily smoothed: tracks the level, destroys the transient.
        kernel = np.ones(9) / 9.0
        smooth = np.convolve(truth, kernel, mode="same")

        m = R.ramp_metrics(truth, smooth, cs)
        assert rmse(truth, smooth) < np.std(truth)
        assert m.recall < 1.0

    def test_base_rate_reported(self):
        cs, truth = self._series()
        m = R.ramp_metrics(truth, truth, cs)
        assert 0.0 < m.base_rate < 1.0

    def test_no_observed_ramps_gives_nan_recall(self):
        cs = np.full(30, 800.0)
        flat = np.full(30, 700.0)
        m = R.ramp_metrics(flat, flat, cs)
        assert m.n_observed == 0
        assert np.isnan(m.recall)
