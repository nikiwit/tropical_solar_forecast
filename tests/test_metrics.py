"""Metric correctness against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from solarfc import metrics as M


class TestPointMetrics:
    def test_perfect_forecast(self):
        y = np.array([100.0, 200.0, 300.0])
        assert M.mae(y, y) == pytest.approx(0.0)
        assert M.rmse(y, y) == pytest.approx(0.0)
        assert M.mbe(y, y) == pytest.approx(0.0)
        assert M.r2(y, y) == pytest.approx(1.0)

    def test_known_values(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 320.0, 380.0])
        # errors: +10, -10, +20, -20
        assert M.mae(y_true, y_pred) == pytest.approx(15.0)
        assert M.rmse(y_true, y_pred) == pytest.approx(np.sqrt(250.0))
        assert M.mbe(y_true, y_pred) == pytest.approx(0.0)

    def test_mbe_sign_is_over_prediction_positive(self):
        y_true = np.array([100.0, 100.0])
        assert M.mbe(y_true, np.array([120.0, 120.0])) == pytest.approx(20.0)
        assert M.mbe(y_true, np.array([80.0, 80.0])) == pytest.approx(-20.0)

    def test_nan_dropped_pairwise(self):
        y_true = np.array([100.0, np.nan, 300.0])
        y_pred = np.array([110.0, 200.0, np.nan])
        assert M.mae(y_true, y_pred) == pytest.approx(10.0)

    def test_all_nan_returns_nan_not_raise(self):
        nan3 = np.full(3, np.nan)
        assert np.isnan(M.mae(nan3, nan3))
        assert np.isnan(M.rmse(nan3, nan3))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            M.mae(np.zeros(3), np.zeros(4))

    def test_normalised_metrics_are_percentages(self):
        y_true = np.full(4, 200.0)
        y_pred = np.full(4, 220.0)
        assert M.nmae(y_true, y_pred) == pytest.approx(10.0)
        assert M.nrmse(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_floor_excludes_near_zero_denominators(self):
        # The 1.0 sample would dominate MAPE without the floor.
        y_true = np.array([1.0, 200.0, 400.0])
        y_pred = np.array([50.0, 220.0, 440.0])
        assert M.mape(y_true, y_pred) == pytest.approx(10.0)


class TestDaytimeMask:
    def test_excludes_night_and_nan(self):
        cs = np.array([0.0, 5.0, 25.0, 800.0, np.nan])
        assert M.daytime_mask(cs).tolist() == [False, False, True, True, False]

    def test_night_inclusion_would_flatter_mae(self):
        """The reason the mask exists, asserted rather than assumed."""
        cs = np.array([0.0, 0.0, 0.0, 500.0, 800.0])
        y_true = np.array([0.0, 0.0, 0.0, 500.0, 800.0])
        y_pred = np.array([0.0, 0.0, 0.0, 400.0, 700.0])

        day = M.daytime_mask(cs)
        assert M.mae(y_true[day], y_pred[day]) == pytest.approx(100.0)
        assert M.mae(y_true, y_pred) == pytest.approx(40.0)


class TestForecastSkill:
    def test_matching_reference_scores_zero(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 310.0])
        assert M.forecast_skill(y_true, y_pred, y_pred) == pytest.approx(0.0)

    def test_perfect_forecast_scores_one(self):
        y_true = np.array([100.0, 200.0, 300.0])
        ref = np.array([150.0, 250.0, 350.0])
        assert M.forecast_skill(y_true, y_true, ref) == pytest.approx(1.0)

    def test_worse_than_reference_is_negative(self):
        y_true = np.array([100.0, 200.0, 300.0])
        good = np.array([105.0, 205.0, 305.0])
        bad = np.array([200.0, 300.0, 400.0])
        assert M.forecast_skill(y_true, bad, good) < 0.0

    def test_perfect_reference_returns_nan(self):
        y_true = np.array([100.0, 200.0])
        assert np.isnan(M.forecast_skill(y_true, np.array([1.0, 2.0]), y_true))


class TestProbabilistic:
    def test_pinball_median_is_half_absolute_error(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([120.0, 180.0])
        assert M.pinball_loss(y_true, y_pred, 0.5) == pytest.approx(10.0)

    def test_pinball_penalises_asymmetrically(self):
        """At q=0.9, under-prediction must cost more than over-prediction."""
        y_true = np.array([100.0])
        under = M.pinball_loss(y_true, np.array([80.0]), 0.9)
        over = M.pinball_loss(y_true, np.array([120.0]), 0.9)
        assert under > over

    def test_invalid_quantile_raises(self):
        with pytest.raises(ValueError, match="quantile must be in"):
            M.pinball_loss([1.0], [1.0], 1.0)

    def test_picp_counts_inclusive_coverage(self):
        y_true = np.array([50.0, 150.0, 250.0, 350.0])
        lower = np.full(4, 100.0)
        upper = np.full(4, 300.0)
        assert M.picp(y_true, lower, upper) == pytest.approx(0.5)

    def test_picp_boundary_counts_as_covered(self):
        assert M.picp([100.0, 300.0], [100.0, 100.0], [300.0, 300.0]) == pytest.approx(1.0)

    def test_pinaw_normalises_by_observed_range(self):
        y_true = np.array([0.0, 1000.0])
        assert M.pinaw(y_true, np.array([0.0, 0.0]), np.array([100.0, 100.0])) == (
            pytest.approx(0.1)
        )

    def test_reliability_curve_detects_overconfidence(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(500.0, 100.0, 20_000)
        # Intervals far too narrow: empirical coverage must fall inside nominal.
        q_pred = np.column_stack(
            [np.full(y_true.size, 480.0), np.full(y_true.size, 520.0)]
        )
        nominal, empirical = M.reliability_curve(y_true, q_pred, [0.1, 0.9])
        assert nominal.tolist() == [0.1, 0.9]
        assert empirical[0] > 0.1  # P10 too high
        assert empirical[1] < 0.9  # P90 too low

    def test_reliability_curve_shape_validation(self):
        with pytest.raises(ValueError, match="column count"):
            M.reliability_curve(np.zeros(5), np.zeros((5, 3)), [0.1, 0.9])


class TestPointMetricsBundle:
    def test_stable_key_set_with_and_without_reference(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 290.0])
        ref = np.array([150.0, 250.0, 350.0])

        without = M.point_metrics(y_true, y_pred)
        with_ref = M.point_metrics(y_true, y_pred, ref)

        assert without.keys() == with_ref.keys()
        assert np.isnan(without["forecast_skill"])
        assert with_ref["forecast_skill"] > 0.0
        assert with_ref["n"] == 3
