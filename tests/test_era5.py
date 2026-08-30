"""ERA5 loading, site extraction, upsampling and unit derivation.

The accumulation-alignment tests are the important ones here. A one-hour
misalignment between ERA5 and NSRDB would not raise an error — it would
quietly degrade every model in the project, which is the kind of bug that is
never found by looking at results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from solarfc import era5 as E
from solarfc.config import SITES_BY_KEY


def _grid(n_hours=48, start="2016-01-01"):
    """Minimal ERA5-shaped dataset over a small SEA grid."""
    time = pd.date_range(start, periods=n_hours, freq="h")
    lat = np.array([5.0, 4.75, 4.5, 4.25, 4.0])
    lon = np.array([100.0, 100.25, 100.5, 100.75, 101.0])
    shape = (n_hours, lat.size, lon.size)

    def field(base):
        return (("valid_time", "latitude", "longitude"), np.full(shape, base))

    return xr.Dataset(
        {
            "tcc": field(0.5),
            "t2m": field(300.0),
            "d2m": field(295.0),
            "msl": field(101_000.0),
            "u10": field(3.0),
            "v10": field(4.0),
            "tcwv": field(45.0),
            "tp": field(0.001),
            "ssrd": field(1_800_000.0),
        },
        coords={"valid_time": time, "latitude": lat, "longitude": lon},
    )


class TestExtractSite:
    def test_nearest_gridpoint_selected(self):
        ds = _grid()
        df = E.extract_site(ds, "penang")  # 5.414 N, 100.330 E
        assert df.attrs["grid_lat"] == pytest.approx(5.0)
        assert df.attrs["grid_lon"] == pytest.approx(100.25)

    def test_offset_recorded_for_the_methodology_chapter(self):
        df = E.extract_site(_grid(), "penang")
        assert 0.0 < df.attrs["offset_km"] < 60.0
        assert df.attrs["site"] == "penang"

    def test_index_is_utc(self):
        df = E.extract_site(_grid(), "penang")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

    def test_grid_coords_not_left_as_columns(self):
        df = E.extract_site(_grid(), "penang")
        assert "latitude" not in df.columns
        assert "longitude" not in df.columns

    def test_all_expected_variables_present(self):
        df = E.extract_site(_grid(), "penang")
        for var in (*E.INSTANT_VARS, *E.ACCUM_VARS):
            assert var in df.columns


class TestUpsampleInstantaneous:
    def test_target_grid_is_10_minutes(self):
        df = E.extract_site(_grid(n_hours=3), "penang")
        out = E.upsample_to_10min(df)
        deltas = out.index.to_series().diff().dropna().dt.total_seconds() / 60
        assert set(deltas.unique()) == {10.0}

    def test_linear_interpolation_between_hourly_samples(self):
        df = E.extract_site(_grid(n_hours=3), "penang")
        df["t2m"] = [300.0, 306.0, 312.0]  # +6 K per hour

        out = E.upsample_to_10min(df)
        # 30 minutes past the first sample should sit exactly halfway.
        halfway = out["t2m"].iloc[3]
        assert halfway == pytest.approx(303.0)

    def test_cloud_ffill_holds_value_constant(self):
        df = E.extract_site(_grid(n_hours=3), "penang")
        df["tcc"] = [0.2, 0.8, 0.4]

        out = E.upsample_to_10min(df, cloud_method="ffill")
        assert out["tcc"].iloc[0] == pytest.approx(0.2)
        assert out["tcc"].iloc[5] == pytest.approx(0.2)  # still within hour 1
        assert out["tcc"].iloc[6] == pytest.approx(0.8)  # steps on the hour

    def test_cloud_linear_interpolates(self):
        df = E.extract_site(_grid(n_hours=3), "penang")
        df["tcc"] = [0.2, 0.8, 0.4]

        out = E.upsample_to_10min(df, cloud_method="linear")
        assert out["tcc"].iloc[3] == pytest.approx(0.5)  # halfway 0.2 -> 0.8

    def test_invalid_cloud_method_raises(self):
        df = E.extract_site(_grid(n_hours=3), "penang")
        with pytest.raises(ValueError, match="cloud_method"):
            E.upsample_to_10min(df, cloud_method="cubic")

    def test_non_datetime_index_raises(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            E.upsample_to_10min(pd.DataFrame({"t2m": [1.0, 2.0]}))


class TestUpsampleAccumulations:
    """ERA5 accumulations cover the hour ENDING at valid_time.

    Empirically confirmed against NSRDB: cross-correlating ERA5 ssrd with
    NSRDB GHI over KL 2016 Q1 peaks at zero lag once the series is shifted
    back by one hour, and at -60 min without the shift.
    """

    def test_accumulation_is_converted_to_a_rate(self):
        df = E.extract_site(_grid(n_hours=3), "penang")
        out = E.upsample_to_10min(df)
        # 1_800_000 J/m^2 over one hour = 500 W/m^2.
        assert out["ssrd_rate"].iloc[0] == pytest.approx(500.0)

    def test_rate_columns_are_renamed(self):
        out = E.upsample_to_10min(E.extract_site(_grid(n_hours=3), "penang"))
        assert "ssrd_rate" in out.columns and "tp_rate" in out.columns
        assert "ssrd" not in out.columns and "tp" not in out.columns

    def test_accumulation_assigned_to_the_window_it_describes(self):
        """The value stamped 01:00 covers 00:00-01:00, so it must land on the
        00:00..00:50 sub-steps, not on 01:00..01:50."""
        df = E.extract_site(_grid(n_hours=3), "penang")
        df["ssrd"] = [0.0, 3_600_000.0, 0.0]  # 1000 W/m^2 during 00:00-01:00

        out = E.upsample_to_10min(df)
        first_hour = out.loc[
            "2016-01-01 00:00":"2016-01-01 00:50", "ssrd_rate"
        ]
        second_hour = out.loc[
            "2016-01-01 01:00":"2016-01-01 01:50", "ssrd_rate"
        ]

        assert np.allclose(first_hour.to_numpy(), 1000.0)
        assert np.allclose(second_hour.to_numpy(), 0.0)

    def test_accumulation_is_held_constant_not_interpolated(self):
        """Interpolating an integral would smear energy across hour boundaries."""
        df = E.extract_site(_grid(n_hours=3), "penang")
        df["ssrd"] = [0.0, 3_600_000.0, 0.0]

        out = E.upsample_to_10min(df)
        within = out.loc["2016-01-01 00:00":"2016-01-01 00:50", "ssrd_rate"]
        assert within.nunique() == 1


class TestDeriveFeatures:
    def _out(self, **overrides):
        df = E.extract_site(_grid(n_hours=3), "penang")
        for k, v in overrides.items():
            df[k] = v
        return E.derive_features(E.upsample_to_10min(df))

    def test_temperature_converted_to_celsius(self):
        assert self._out()["era5_temp_c"].iloc[0] == pytest.approx(26.85)

    def test_pressure_converted_to_hpa(self):
        assert self._out()["era5_pressure_hpa"].iloc[0] == pytest.approx(
            1010.0
        )

    def test_wind_speed_from_components(self):
        # u=3, v=4 -> 5
        assert self._out()["era5_wind_speed"].iloc[0] == pytest.approx(5.0)

    def test_wind_direction_is_the_direction_it_blows_from(self):
        """Meteorological convention. A southerly wind (v>0, blowing north)
        must report ~180 degrees, not 0."""
        out = self._out(u10=0.0, v10=5.0)
        assert out["era5_wind_direction"].iloc[0] == pytest.approx(180.0)

    def test_westerly_wind_direction(self):
        out = self._out(u10=5.0, v10=0.0)  # blowing east, comes from the west
        assert out["era5_wind_direction"].iloc[0] == pytest.approx(270.0)

    def test_precipitation_converted_to_mm_per_hour(self):
        # tp = 0.001 m accumulated in one hour = 1 mm/h
        assert self._out()["era5_precip_mm_h"].iloc[0] == pytest.approx(1.0)

    def test_relative_humidity_derived_and_bounded(self):
        rh = self._out()["era5_relative_humidity"]
        assert 0.0 <= rh.iloc[0] <= 100.0
        # 300 K air, 295 K dewpoint -> humid but not saturated.
        assert 60.0 < rh.iloc[0] < 90.0

    def test_saturated_air_gives_100_percent(self):
        out = self._out(d2m=300.0)  # dewpoint == temperature
        assert out["era5_relative_humidity"].iloc[0] == pytest.approx(
            100.0, abs=0.5
        )

    def test_drier_air_lowers_humidity(self):
        humid = self._out(d2m=299.0)["era5_relative_humidity"].iloc[0]
        dry = self._out(d2m=285.0)["era5_relative_humidity"].iloc[0]
        assert dry < humid


class TestOpenMonth:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="ERA5 file not found"):
            E.open_month("/nonexistent/era5_sea_2016_01.nc")
