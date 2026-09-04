"""Bounds on forecast error, and the properties that keep them bounds."""

from __future__ import annotations

import numpy as np
import pytest

from solarfc import predictability as P


class TestAutocorrelation:
    def test_recovers_a_known_ar1_coefficient(self):
        rng = np.random.default_rng(0)
        n, phi = 200_000, 0.8
        x = np.zeros(n)
        noise = rng.normal(0, 1, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + noise[i]
        rho, pairs = P.autocorrelation(x, 1)
        assert rho == pytest.approx(phi, abs=0.01)
        assert pairs == n - 1

    def test_ar1_decays_geometrically(self):
        rng = np.random.default_rng(1)
        n, phi = 200_000, 0.8
        x = np.zeros(n)
        noise = rng.normal(0, 1, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + noise[i]
        rho3, _ = P.autocorrelation(x, 3)
        assert rho3 == pytest.approx(phi**3, abs=0.02)

    def test_gaps_are_skipped_not_closed_up(self):
        """The whole point of holding night as NaN.

        99 lag-1 pairs exist over 100 samples. A hole at 50..59 kills
        every pair whose either end lands in it, so i = 49..59, eleven
        of them. The pairs either side of the hole are simply dropped
        rather than joined across it.
        """
        x = np.arange(100.0)
        x[50:60] = np.nan
        rho, pairs = P.autocorrelation(x, 1)
        assert pairs == 99 - 11

    def test_too_few_pairs_gives_nan(self):
        """A 12 h lag on a daytime-masked series has no pairs at all."""
        x = np.full(500, np.nan)
        x[:10] = 1.0
        rho, pairs = P.autocorrelation(x, 1)
        assert np.isnan(rho)
        assert pairs < P.MIN_PAIRS

    def test_rejects_a_zero_lag(self):
        with pytest.raises(ValueError, match="lag_steps"):
            P.autocorrelation(np.zeros(10), 0)


class TestCauchyCorrelogram:
    def test_is_one_at_the_origin_and_the_nugget_just_after(self):
        c = P.CauchyCorrelogram(
            nugget=0.1, scale_hours=1.0, alpha=1.0, beta=1.0
        )
        assert c(0.0) == pytest.approx(1.0)
        assert c(1e-9) == pytest.approx(0.9, abs=1e-6)

    def test_decreases_monotonically(self):
        c = P.CauchyCorrelogram(
            nugget=0.05, scale_hours=2.0, alpha=1.5, beta=1.0
        )
        tau = np.linspace(0.01, 100, 400)
        assert np.all(np.diff(c(tau)) <= 1e-12)

    def test_stays_within_zero_and_one(self):
        c = P.CauchyCorrelogram(
            nugget=0.2, scale_hours=0.5, alpha=2.0, beta=0.5
        )
        v = c(np.linspace(0, 500, 500))
        assert v.min() >= 0.0 and v.max() <= 1.0

    def test_nugget_in_rmse_terms_grows_with_the_nugget(self):
        small = P.CauchyCorrelogram(0.05, 1.0, 1.0, 1.0).nugget_rmse(
            0.09, 4.6e5
        )
        large = P.CauchyCorrelogram(0.20, 1.0, 1.0, 1.0).nugget_rmse(
            0.09, 4.6e5
        )
        assert 0 < small < large


class TestFitCorrelogram:
    def test_recovers_known_parameters(self):
        truth = P.CauchyCorrelogram(
            nugget=0.08, scale_hours=0.5, alpha=1.6, beta=0.7
        )
        tau = np.concatenate([np.arange(1, 7) / 6.0, np.arange(1, 103, 1.0)])
        fitted = P.fit_correlogram(tau, truth(tau))
        np.testing.assert_allclose(fitted(tau), truth(tau), atol=0.01)

    def test_sub_hourly_lags_are_what_identify_the_nugget(self):
        """Fitting from 1 h up leaves the nugget on the boundary."""
        truth = P.CauchyCorrelogram(
            nugget=0.08, scale_hours=0.5, alpha=1.6, beta=0.7
        )
        hourly = np.arange(1, 103, 1.0)
        dense = np.concatenate([np.arange(1, 7) / 6.0, hourly])
        from_hourly = P.fit_correlogram(hourly, truth(hourly))
        from_dense = P.fit_correlogram(dense, truth(dense))
        assert abs(from_dense.nugget - 0.08) < abs(from_hourly.nugget - 0.08)

    def test_ignores_lags_with_no_valid_pairs(self):
        truth = P.CauchyCorrelogram(0.08, 0.5, 1.6, 0.7)
        tau = np.concatenate([np.arange(1, 7) / 6.0, np.arange(1, 103, 1.0)])
        rho = truth(tau)
        rho[(tau == 12.0) | (tau == 36.0)] = np.nan
        fitted = P.fit_correlogram(tau, rho)
        assert fitted.nugget == pytest.approx(0.08, abs=0.02)

    def test_refuses_to_fit_on_too_few_lags(self):
        with pytest.raises(ValueError, match="usable lags"):
            P.fit_correlogram([1.0, 2.0], [0.5, 0.4])


class TestUpperBound:
    def test_rises_with_horizon_and_saturates(self):
        c = P.CauchyCorrelogram(0.05, 0.5, 2.0, 0.6)
        a = P.upper_bound_rmse(c, [0.17, 1, 3, 6, 24, 48], 0.092, 4.6e5)
        assert np.all(np.diff(a) > 0)
        assert a[-1] - a[-2] < a[1] - a[0]

    def test_is_zero_at_zero_lead(self):
        """C(0) = 1, so the reference is exact and the bound vanishes."""
        c = P.CauchyCorrelogram(0.05, 0.5, 2.0, 0.6)
        assert P.upper_bound_rmse(c, 0.0, 0.092, 4.6e5) == pytest.approx(0.0)

    def test_scales_with_the_clear_sky_envelope(self):
        c = P.CauchyCorrelogram(0.05, 0.5, 2.0, 0.6)
        a = P.upper_bound_rmse(c, 6.0, 0.092, 4.6e5)
        b = P.upper_bound_rmse(c, 6.0, 0.092, 4.0 * 4.6e5)
        assert b == pytest.approx(2.0 * a)


class TestMspeg:
    def test_is_zero_when_members_match_the_control(self):
        control = np.linspace(0, 1, 50)
        assert P.mspeg(control, np.tile(control[:, None], (1, 10))) == 0.0

    def test_equals_the_mean_square_offset(self):
        control = np.zeros(100)
        assert P.mspeg(control, np.full((100, 5), 3.0)) == pytest.approx(9.0)

    def test_accepts_a_single_member(self):
        assert P.mspeg(np.zeros(10), np.full(10, 2.0)) == pytest.approx(4.0)

    def test_rejects_misaligned_members(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            P.mspeg(np.zeros(10), np.zeros((9, 3)))


class TestErrorGrowthAndPredictability:
    def _correlogram(self):
        return P.CauchyCorrelogram(0.08, 0.5, 2.0, 0.6)

    def test_recovers_a_known_slope(self):
        c = self._correlogram()
        var_k = 0.092
        intercept = (1 - (1 - c.nugget) ** 2) * var_k
        tau = np.array([6.0, 12, 24, 48, 96])
        fitted = P.fit_error_growth(
            tau, 0.004 * tau + intercept, correlogram=c, variance_kappa=var_k
        )
        assert fitted.slope == pytest.approx(0.004, rel=1e-6)

    def test_intercept_comes_from_the_nugget_not_the_data(self):
        """Eq. 8 fixes it, which is why A_r must be fitted first."""
        c = self._correlogram()
        var_k = 0.092
        tau = np.array([6.0, 24, 48])
        fitted = P.fit_error_growth(
            tau, np.zeros_like(tau), correlogram=c, variance_kappa=var_k
        )
        assert fitted.intercept() == pytest.approx(
            (1 - (1 - 0.08) ** 2) * var_k
        )

    def test_lower_bound_grows_with_horizon(self):
        c = self._correlogram()
        g = P.fit_error_growth(
            np.array([6.0, 24, 48]),
            np.array([0.05, 0.12, 0.22]),
            correlogram=c,
            variance_kappa=0.092,
        )
        a = g.lower_bound_rmse([1, 6, 24, 48], 4.6e5)
        assert np.all(np.diff(a) > 0)

    def test_predictability_is_between_zero_and_one(self):
        assert P.predictability(50.0, 200.0) == pytest.approx(0.75)
        assert P.predictability(0.0, 200.0) == pytest.approx(1.0)
        assert P.predictability(200.0, 200.0) == pytest.approx(0.0)

    def test_a_lower_bound_above_the_upper_bound_clips_to_zero(self):
        """Both are estimates, so crossing is possible and means no skill."""
        assert P.predictability(300.0, 200.0) == pytest.approx(0.0)
