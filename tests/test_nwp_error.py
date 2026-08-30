"""NWP forecast-error measurement and degradation-model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarfc import nwp_error as N
from solarfc.covariates import ErrorModel


def _jma(n=2000, seed=0):
    """Synthetic JMA-shaped frame with known, lead-growing error."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    truth = 0.5 + 0.2 * np.sin(np.arange(n) / 24.0)

    frame = pd.DataFrame(index=idx)
    frame["era5_cloud_cover_lead0h"] = truth
    for lead, sigma in ((24, 0.10), (48, 0.20), (72, 0.30)):
        frame[f"era5_cloud_cover_lead{lead}h"] = truth + rng.normal(0, sigma, n)
    return frame


def _era5(index, offset=0.0):
    n = len(index)
    return pd.DataFrame(
        {"era5_cloud_cover": 0.5 + 0.2 * np.sin(np.arange(n) / 24.0) + offset},
        index=index,
    )


class TestErrorSeries:
    def test_drift_mode_differences_against_lead_zero(self):
        jma = _jma()
        series = N.error_series(jma, "era5_cloud_cover", 24, mode="drift")
        expected = (
            jma["era5_cloud_cover_lead24h"] - jma["era5_cloud_cover_lead0h"]
        )
        np.testing.assert_allclose(series.to_numpy(), expected.to_numpy())

    def test_total_mode_differences_against_era5(self):
        jma = _jma()
        era5 = _era5(jma.index)
        series = N.error_series(
            jma, "era5_cloud_cover", 24, mode="total", era5=era5
        )
        assert len(series) == len(jma)

    def test_total_mode_requires_era5(self):
        with pytest.raises(ValueError, match="requires the era5 frame"):
            N.error_series(_jma(), "era5_cloud_cover", 24, mode="total")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            N.error_series(_jma(), "era5_cloud_cover", 24, mode="rmse")

    def test_missing_column_returns_none(self):
        assert N.error_series(_jma(), "era5_temp_c", 24, mode="drift") is None

    def test_drift_at_lead_zero_is_identically_zero(self):
        series = N.error_series(_jma(), "era5_cloud_cover", 0, mode="drift")
        assert np.allclose(series.to_numpy(), 0.0)

    def test_total_captures_bias_that_drift_cancels(self):
        """The whole reason total is used to calibrate the realistic track."""
        jma = _jma()
        era5 = _era5(jma.index, offset=0.15)  # ERA5 systematically higher

        drift = N.measure_error(jma, "era5_cloud_cover", 24, mode="drift")
        total = N.measure_error(
            jma, "era5_cloud_cover", 24, mode="total", era5=era5
        )
        assert abs(drift.bias) < 0.02
        assert total.bias == pytest.approx(-0.15, abs=0.02)


class TestMeasureError:
    def test_recovers_known_sigma(self):
        stats = N.measure_error(_jma(), "era5_cloud_cover", 24, mode="drift")
        assert stats.std == pytest.approx(0.10, abs=0.01)

    def test_error_grows_with_lead(self):
        jma = _jma()
        stds = [
            N.measure_error(jma, "era5_cloud_cover", lead, mode="drift").std
            for lead in (24, 48, 72)
        ]
        assert stds[0] < stds[1] < stds[2]

    def test_reports_sample_count(self):
        stats = N.measure_error(_jma(n=500), "era5_cloud_cover", 24, mode="drift")
        assert stats.n == 500


class TestFitErrorModel:
    def test_recovers_growth_rate(self):
        model, measured = N.fit_error_model(
            _jma(n=20_000), "era5_cloud_cover", mode="drift", bounds=(0.0, 1.0)
        )
        # sigma goes 0.10 -> 0.30 over 24 -> 72 h, so ~0.00417 per hour.
        assert model.growth_rate == pytest.approx(0.00417, rel=0.15)

    def test_bounds_carried_onto_the_model(self):
        model, _ = N.fit_error_model(
            _jma(), "era5_cloud_cover", mode="drift", bounds=(0.0, 1.0)
        )
        assert model.lower_bound == 0.0 and model.upper_bound == 1.0

    def test_sigma_zero_is_positive(self):
        """A negative intercept is physically meaningless."""
        model, _ = N.fit_error_model(_jma(), "era5_cloud_cover", mode="drift")
        assert model.sigma_0 > 0.0

    def test_too_few_leads_raises(self):
        jma = _jma()[["era5_cloud_cover_lead0h", "era5_cloud_cover_lead24h"]]
        thin = jma.drop(columns=["era5_cloud_cover_lead24h"])
        with pytest.raises(ValueError, match="at least 2 usable lead times"):
            N.fit_error_model(thin, "era5_cloud_cover", mode="drift")


class TestCorrelationFitting:
    def test_persistent_error_gives_long_correlation(self):
        n = 5000
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(0)

        # AR(1) error with a long timescale.
        phi = np.exp(-1.0 / 12.0)
        e = np.empty(n)
        e[0] = rng.standard_normal()
        for i in range(1, n):
            e[i] = phi * e[i - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()

        frame = pd.DataFrame(index=idx)
        frame["v_lead0h"] = 0.5
        frame["v_lead24h"] = 0.5 + 0.1 * e

        autocorr = N.measure_error_autocorrelation(frame, "v", 24)
        assert N._fit_correlation_hours(autocorr) > 6.0

    def test_missing_columns_give_empty_autocorrelation(self):
        assert N.measure_error_autocorrelation(_jma(), "absent", 24).empty

    def test_fallback_when_autocorrelation_unusable(self):
        assert N._fit_correlation_hours(pd.Series(dtype=float)) == 6.0


class TestValidateFit:
    def test_good_fit_passes(self):
        model, measured = N.fit_error_model(
            _jma(n=20_000), "era5_cloud_cover", mode="drift"
        )
        report = N.validate_fit(model, measured)
        assert report["within_tolerance"].all()

    def test_bad_fit_is_flagged_not_raised(self):
        """The caller decides what to do about a failure."""
        _, measured = N.fit_error_model(_jma(), "era5_cloud_cover", mode="drift")
        wrong = ErrorModel("era5_cloud_cover", 5.0, 0.0, 10.0)
        report = N.validate_fit(wrong, measured)
        assert not report["within_tolerance"].any()

    def test_lead_zero_excluded_from_the_report(self):
        model, measured = N.fit_error_model(
            _jma(), "era5_cloud_cover", mode="drift"
        )
        report = N.validate_fit(model, measured)
        assert (report["lead_hours"] > 0).all()


class TestErrorModelApplication:
    def test_bias_shifts_the_field(self):
        from solarfc.covariates import degrade

        model = ErrorModel("v", sigma_0=0.0, growth_rate=0.0, sigma_max=0.0,
                           bias_0=-0.2)
        out = degrade(np.full(100, 0.5), model, np.full(100, 24.0),
                      rng=np.random.default_rng(0))
        assert out.mean() == pytest.approx(0.3, abs=1e-9)

    def test_bias_grows_with_lead(self):
        model = ErrorModel("v", 0.1, 0.0, 1.0, bias_0=0.0, bias_growth=0.01)
        assert model.bias_at(0.0) == pytest.approx(0.0)
        assert model.bias_at(48.0) == pytest.approx(0.48)

    def test_bounds_applied_after_bias(self):
        from solarfc.covariates import degrade

        model = ErrorModel("v", 0.0, 0.0, 0.0, bias_0=-5.0, lower_bound=0.0)
        out = degrade(np.full(50, 0.5), model, np.full(50, 24.0),
                      rng=np.random.default_rng(0))
        assert out.min() >= 0.0
