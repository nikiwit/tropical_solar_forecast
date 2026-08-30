"""Reference forecasts, and the leakage guards that keep them honest."""

from __future__ import annotations

import numpy as np
import pytest

from solarfc import baselines as B


class TestBaselineLeakage:
    """A forecast must never use information from at or after its target time."""

    def test_naive_persistence_uses_only_past(self):
        ghi = np.arange(100, dtype=float)
        pred = B.naive_persistence(ghi, horizon_steps=6)
        assert np.all(np.isnan(pred[:6]))
        # Forecast for index i is the observation from i-6.
        assert pred[10] == pytest.approx(ghi[4])

    def test_naive_persistence_cannot_see_the_target(self):
        """Mutating only the future must not change any earlier forecast."""
        ghi = np.arange(100, dtype=float)
        before = B.naive_persistence(ghi, horizon_steps=6)

        tampered = ghi.copy()
        tampered[50:] = 9999.0
        after = B.naive_persistence(tampered, horizon_steps=6)

        np.testing.assert_allclose(before[:56], after[:56], equal_nan=True)

    def test_smart_persistence_uses_only_past_ghi(self):
        """CSI must come from the origin; only clear-sky may come from the target."""
        n = 100
        cs = np.full(n, 800.0)
        ghi = np.full(n, 400.0)

        before = B.smart_persistence(ghi, cs, horizon_steps=6)
        tampered = ghi.copy()
        tampered[50:] = 0.0
        after = B.smart_persistence(tampered, cs, horizon_steps=6)

        np.testing.assert_allclose(before[:56], after[:56], equal_nan=True)

    def test_carry_overnight_does_not_leak_future(self):
        """Forward fill must look backwards only."""
        cs = np.tile(np.concatenate([np.full(20, 800.0), np.zeros(20)]), 5)
        ghi = cs * 0.5

        before = B.smart_persistence(ghi, cs, 12)
        tampered = ghi.copy()
        tampered[100:] = 0.0
        after = B.smart_persistence(tampered, cs, 12)

        np.testing.assert_allclose(before[:112], after[:112], equal_nan=True)

    def test_zero_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon_steps"):
            B.naive_persistence(np.zeros(10), 0)


class TestSmartPersistence:
    def test_beats_naive_across_the_diurnal_cycle(self):
        """The reason smart persistence is the primary FS reference."""
        from solarfc.metrics import rmse

        n = 144  # one day at 10-minute resolution
        t = np.arange(n)
        cs = np.maximum(0.0, 1000.0 * np.sin(np.pi * t / n))
        ghi = cs * 0.7  # constant CSI: a steady hazy day

        horizon = 36  # 6 hours
        naive = B.naive_persistence(ghi, horizon)
        smart = B.smart_persistence(ghi, cs, horizon)

        ok = np.isfinite(naive) & np.isfinite(smart)
        assert rmse(ghi[ok], smart[ok]) < rmse(ghi[ok], naive[ok])

    def test_night_targets_forecast_zero(self):
        cs = np.concatenate([np.full(50, 800.0), np.zeros(50)])
        ghi = np.concatenate([np.full(50, 400.0), np.zeros(50)])
        pred = B.smart_persistence(ghi, cs, horizon_steps=6)
        assert np.all(pred[50:] == 0.0)

    def test_defined_when_origin_falls_at_night(self):
        """At 12h every daytime target has a night origin; carrying CSI
        overnight keeps the FS reference defined instead of NaN."""
        day = np.full(72, 800.0)
        night = np.zeros(72)
        cs = np.concatenate([day, night, day])
        ghi = np.concatenate([day * 0.6, night, day * 0.6])

        carried = B.smart_persistence(ghi, cs, 72, carry_overnight=True)
        strict = B.smart_persistence(ghi, cs, 72, carry_overnight=False)

        targets = slice(144, 216)  # the second daytime block
        assert np.all(np.isfinite(carried[targets]))
        assert np.all(np.isnan(strict[targets]))
        # Carried CSI (0.6) rescaled by clear-sky at the target time.
        assert carried[150] == pytest.approx(480.0)

    def test_forward_fill_leaves_leading_nans(self):
        out = B._forward_fill(np.array([np.nan, np.nan, 0.5, np.nan, np.nan]))
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert out[2] == pytest.approx(0.5)
        assert out[3] == pytest.approx(0.5)
        assert out[4] == pytest.approx(0.5)


class TestClearSkyIndex:
    def test_ratio_computed_in_daylight(self):
        csi = B.clear_sky_index(np.array([400.0]), np.array([800.0]))
        assert csi[0] == pytest.approx(0.5)

    def test_nan_at_night_not_division_blowup(self):
        csi = B.clear_sky_index(np.array([0.0, 5.0]), np.array([0.0, 10.0]))
        assert np.all(np.isnan(csi))

    def test_cloud_enhancement_preserved_above_unity(self):
        """Clipping at 1.0 would erase the over-irradiance events ramp analysis needs."""
        csi = B.clear_sky_index(np.array([960.0]), np.array([800.0]))
        assert csi[0] == pytest.approx(1.2)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            B.clear_sky_index(np.zeros(5), np.ones(6))


class TestClearSkyForecast:
    def test_returns_the_envelope_itself(self):
        cs = np.array([0.0, 400.0, 900.0])
        np.testing.assert_allclose(B.clear_sky_forecast(cs), cs)

    def test_does_not_alias_the_input(self):
        cs = np.array([100.0, 200.0])
        out = B.clear_sky_forecast(cs)
        out[0] = -1.0
        assert cs[0] == pytest.approx(100.0)
