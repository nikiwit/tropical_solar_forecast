"""Supervised-matrix assembly, and the alignment it depends on.

:mod:`solarfc.dataset` is the only module in the package that shifts a series
forwards, so it is the only place a leakage bug can start. The tests here assert
the alignment numerically in both directions rather than checking that the code
reads correctly:

* a known-future column on row ``t`` must equal the unshifted value at ``t + h``;
* the target on row ``t`` must equal the observation at ``t + h``;
* perturbing observations after a cut must leave every row before it unchanged.

The last one is the test that would actually catch a sign error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarfc import dataset as D
from solarfc.covariates import NWP_FEATURES
from solarfc.features import build_observed_past

from test_features import _era5, _nsrdb


HORIZON = 6


@pytest.fixture(scope="module")
def raw():
    nsrdb = _nsrdb(seed=3)
    return nsrdb, _era5(nsrdb.index)


@pytest.fixture(scope="module")
def features(raw):
    nsrdb, era5 = raw
    return build_observed_past(nsrdb, "kuala_lumpur", era5=era5)


def _build(features, era5, **kwargs):
    options = dict(
        track="perfect", feature_set="full", target="ghi", era5=era5, drop_night=False
    )
    options.update(kwargs)
    return D.build_supervised(features, "kuala_lumpur", HORIZON, **options)


class TestAlignment:
    def test_target_is_the_future_observation(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        stamp = built.X.index[50]
        expected = features["ghi"].loc[stamp + pd.Timedelta(minutes=10 * HORIZON)]
        assert built.y.loc[stamp] == pytest.approx(expected)

    def test_csi_target_is_the_future_index(self, features, raw):
        _, era5 = raw
        built = _build(features, era5, target="csi")
        stamp = built.X.index[50]
        expected = features["csi"].loc[stamp + pd.Timedelta(minutes=10 * HORIZON)]
        assert built.y.loc[stamp] == pytest.approx(expected)

    def test_known_future_is_read_at_the_target_time(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        stamp = built.X.index[50]
        target_time = stamp + pd.Timedelta(minutes=10 * HORIZON)
        reference = D.build_known_future_grid(
            features.index, "kuala_lumpur", "perfect", era5=era5
        )
        assert built.X.loc[stamp, "kf_clearsky_ghi"] == pytest.approx(
            reference.loc[target_time, "kf_clearsky_ghi"]
        )

    def test_clearsky_series_is_the_target_time_envelope(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        stamp = built.X.index[50]
        expected = features["clearsky_ghi_ineichen"].loc[
            stamp + pd.Timedelta(minutes=10 * HORIZON)
        ]
        assert built.clearsky_ghi.loc[stamp] == pytest.approx(expected)

    def test_observed_past_is_read_at_the_origin(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        stamp = built.X.index[50]
        assert built.X.loc[stamp, "ghi"] == pytest.approx(features["ghi"].loc[stamp])


class TestLeakage:
    def test_perturbing_the_future_leaves_earlier_rows_untouched(self, raw):
        """The decisive test. A sign error on the shift would fail here."""
        nsrdb, era5 = raw
        clean = build_observed_past(nsrdb, "kuala_lumpur", era5=era5)
        before = _build(clean, era5)

        tampered_raw = nsrdb.copy()
        cut = 400
        tampered_raw.iloc[cut:, tampered_raw.columns.get_loc("GHI")] = 9999.0
        tampered = build_observed_past(tampered_raw, "kuala_lumpur", era5=era5)
        after = _build(tampered, era5)

        stamps = before.X.index[before.X.index < nsrdb.index[cut - HORIZON]]
        assert len(stamps)
        np.testing.assert_allclose(
            before.X.loc[stamps].to_numpy(dtype=float),
            after.X.loc[stamps].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_nwp_free_track_carries_no_atmosphere(self, features, raw):
        _, era5 = raw
        built = _build(features, era5, track="nwp_free")
        assert D.nwp_columns_in(built.X) == []

    def test_nwp_tracks_carry_atmosphere(self, features, raw):
        _, era5 = raw
        for track in ("realistic", "perfect"):
            built = _build(features, era5, track=track)
            assert len(D.nwp_columns_in(built.X)) == len(NWP_FEATURES)

    def test_no_bare_future_irradiance_column(self, features, raw):
        """Observed GHI is neither deterministic nor forecast, so it must not
        appear on the known-future side under any track."""
        _, era5 = raw
        for track in D.TRACKS:
            built = _build(features, era5, track=track)
            assert "kf_ghi" not in built.X.columns
            assert "kf_csi" not in built.X.columns

    def test_target_is_not_among_the_features(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        assert "y" not in built.X.columns


class TestTracks:
    def test_realistic_differs_from_perfect(self, features, raw):
        _, era5 = raw
        perfect = _build(features, era5, track="perfect")
        realistic = _build(features, era5, track="realistic")
        assert not np.allclose(
            perfect.X["kf_era5_cloud_cover"].to_numpy(),
            realistic.X["kf_era5_cloud_cover"].to_numpy(),
        )

    def test_realistic_is_reproducible(self, features, raw):
        """Two builds of the same configuration must be bit-identical.

        The degradation is a random draw. If it were not seeded, rerunning an
        experiment would move the numbers by an amount easily mistaken for a
        real effect.
        """
        _, era5 = raw
        first = _build(features, era5, track="realistic")
        second = _build(features, era5, track="realistic")
        np.testing.assert_array_equal(
            first.X["kf_era5_cloud_cover"].to_numpy(),
            second.X["kf_era5_cloud_cover"].to_numpy(),
        )

    def test_seed_varies_across_configurations(self):
        seeds = {
            D.track_seed("kuala_lumpur", "realistic", 6),
            D.track_seed("kuala_lumpur", "realistic", 36),
            D.track_seed("penang", "realistic", 6),
        }
        assert len(seeds) == 3

    def test_unknown_track_raises(self, features, raw):
        _, era5 = raw
        with pytest.raises(ValueError, match="track"):
            _build(features, era5, track="wishful")

    def test_unknown_target_raises(self, features, raw):
        _, era5 = raw
        with pytest.raises(ValueError, match="target"):
            _build(features, era5, target="megawatts")


class TestNightHandling:
    def test_drop_night_removes_night_targets(self, features, raw):
        from solarfc.config import DAYTIME_CLEARSKY_FLOOR

        _, era5 = raw
        built = _build(features, era5, drop_night=True)
        assert (built.clearsky_ghi.to_numpy() > DAYTIME_CLEARSKY_FLOOR).all()

    def test_drop_night_keeps_night_origins(self, features, raw):
        """A 6 h forecast for 08:00 is issued at 02:00. Keeping that row is the
        whole reason CSI is carried across the night."""
        _, era5 = raw
        built = _build(features, era5, drop_night=True)
        origin_clearsky = features["clearsky_ghi_ineichen"].reindex(built.X.index)
        assert (origin_clearsky.to_numpy() <= 20.0).any()

    def test_keeping_night_yields_more_rows(self, features, raw):
        _, era5 = raw
        assert len(_build(features, era5, drop_night=False)) > len(
            _build(features, era5, drop_night=True)
        )


class TestFeatureSets:
    def test_deployable_is_narrower(self, features, raw):
        _, era5 = raw
        full = _build(features, era5, feature_set="full")
        deployable = _build(features, era5, feature_set="deployable")
        assert set(deployable.X.columns) < set(full.X.columns)

    def test_deployable_keeps_the_nwp_forecast(self, features, raw):
        """A plant can subscribe to a weather forecast; it cannot own a
        satellite. NWP fields therefore stay in the deployable set."""
        _, era5 = raw
        built = _build(features, era5, feature_set="deployable", track="realistic")
        assert len(D.nwp_columns_in(built.X)) == len(NWP_FEATURES)


class TestToGhi:
    def test_ghi_target_passes_through(self):
        values = np.array([100.0, 200.0])
        out = D.to_ghi(values, np.array([500.0, 500.0]), "ghi")
        np.testing.assert_array_equal(out, values)

    def test_csi_target_is_rescaled(self):
        out = D.to_ghi(np.array([0.5, 0.8]), np.array([800.0, 1000.0]), "csi")
        np.testing.assert_allclose(out, [400.0, 800.0])

    def test_negative_predictions_are_clipped(self):
        """A tree can extrapolate below zero; irradiance cannot go there."""
        out = D.to_ghi(np.array([-0.2]), np.array([800.0]), "csi")
        assert out[0] == 0.0

    def test_round_trip_recovers_observed_ghi(self, features, raw):
        """Exact wherever the index was not clipped, which is the normal case.

        The clip is the only thing that makes the rescale lossy, and on real
        data it binds on 0.16% of samples once the clear-sky envelope is
        calibrated. Asserting exactness on the unclipped majority is the real
        property; asserting it everywhere would be asserting the clip away.
        """
        from solarfc.config import CSI_CLIP_MAX

        _, era5 = raw
        built = _build(features, era5, target="csi", drop_night=True)
        recovered = D.to_ghi(built.y.to_numpy(), built.clearsky_ghi.to_numpy(), "csi")
        unclipped = built.y.to_numpy() < CSI_CLIP_MAX
        assert unclipped.any()
        np.testing.assert_allclose(
            recovered[unclipped], built.ghi.to_numpy()[unclipped], rtol=1e-9
        )

    def test_clipping_only_ever_understates(self, features, raw):
        """A clipped sample must come back below the observation, never above."""
        from solarfc.config import CSI_CLIP_MAX

        _, era5 = raw
        built = _build(features, era5, target="csi", drop_night=True)
        recovered = D.to_ghi(built.y.to_numpy(), built.clearsky_ghi.to_numpy(), "csi")
        clipped = built.y.to_numpy() >= CSI_CLIP_MAX
        if clipped.any():
            assert (recovered[clipped] <= built.ghi.to_numpy()[clipped] + 1e-9).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            D.to_ghi(np.array([0.5]), np.array([800.0, 900.0]), "csi")


class TestPrebuiltKnownFuture:
    """Reusing a grid must give bit-identical output to rebuilding it."""

    def test_matches_the_rebuilt_grid(self, features, raw):
        _, era5 = raw
        grid = D.build_known_future_grid(
            features.index, "kuala_lumpur", "perfect", era5=era5
        )
        supplied = _build(features, era5, known_future=grid)
        rebuilt = _build(features, era5)
        pd.testing.assert_frame_equal(supplied.X, rebuilt.X)

    def test_rejects_a_grid_from_another_track(self, features, raw):
        _, era5 = raw
        grid = D.build_known_future_grid(
            features.index, "kuala_lumpur", "nwp_free", era5=era5
        )
        with pytest.raises(ValueError, match="does not match track"):
            _build(features, era5, track="perfect", known_future=grid)

    def test_accepts_a_matching_nwp_free_grid(self, features, raw):
        _, era5 = raw
        grid = D.build_known_future_grid(
            features.index, "kuala_lumpur", "nwp_free", era5=era5
        )
        built = _build(features, era5, track="nwp_free", known_future=grid)
        assert D.nwp_columns_in(built.X) == []


class TestSupervisedSet:
    def test_split_subset_selects_by_year(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        # The synthetic fixture is entirely 2020, which is the test year.
        assert len(built.subset("test")) == len(built)
        assert len(built.subset("train")) == 0

    def test_all_series_share_an_index(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        for series in (built.y, built.clearsky_ghi, built.ghi, built.split):
            pd.testing.assert_index_equal(series.index, built.X.index)

    def test_attrs_record_the_configuration(self, features, raw):
        _, era5 = raw
        built = _build(features, era5, track="realistic", feature_set="deployable")
        assert built.X.attrs["track"] == "realistic"
        assert built.X.attrs["feature_set"] == "deployable"
        assert built.X.attrs["lead_hours"] == pytest.approx(1.0)

    def test_no_nan_survives(self, features, raw):
        _, era5 = raw
        built = _build(features, era5)
        assert built.X.notna().all().all()
        assert built.y.notna().all()
