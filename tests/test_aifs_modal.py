"""Comprehensive test suite for aifs_modal.

Avoids Modal, S3, and other infrastructure by:

* mocking the ``modal`` package at import time (see ``conftest.py``),
* using an in-process ``icechunk.local_filesystem_storage`` repo for any code
  that needs an icechunk Repository (forecast outputs, ``forecast_exists``),
* injecting fake runners/regridders into ``_run_inference_impl`` and recorder
  callables into ``_run_forecast_impl``,
* monkeypatching ``_regrid_n320`` / ``_open_arco`` for ingest tests so no
  remote data and no real regridding weights are needed.
"""

from __future__ import annotations

import datetime as dt
import os
from unittest import mock

import icechunk
import numpy as np
import pytest
import xarray as xr

from aifs_modal import (
    _app,
    _ic,
    _ingest_ekd,
    _ingest_era5_arco,
    _ingest_ifs_arraylake,
    settings,
    utils,
)


# ===========================================================================
# Pure helpers in _app.py
# ===========================================================================
class TestRequireOutputsTarget:
    def test_ok_with_prefix(self):
        _app._require_outputs_target(None, "some/prefix")

    def test_ok_with_repo(self):
        _app._require_outputs_target("org/repo", None)

    def test_raises_when_both_none(self):
        with pytest.raises(ValueError, match="outputs_prefix"):
            _app._require_outputs_target(None, None)


class TestIcDatesPresent:
    def test_present_when_both_complete(self, ic_dir, init_date, write_ic_dates):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        assert _app._ic_dates_present(init_date, ic_dir)

    def test_missing_when_only_one_exists(self, ic_dir, init_date, write_ic_dates):
        write_ic_dates([init_date])
        assert not _app._ic_dates_present(init_date, ic_dir)

    def test_empty_dirs_not_present(self, tmp_path, init_date):
        # bare date dirs (e.g. left by a crashed ingest) must not count
        for d in [init_date - dt.timedelta(hours=6), init_date]:
            (tmp_path / utils.datetime_to_str(d)).mkdir(parents=True)
        assert not _app._ic_dates_present(init_date, str(tmp_path))


class TestStateToXarray:
    def _state(self, init_date, fields):
        # state["date"] gets `.replace(tzinfo=None)`'d inside state_to_xarray;
        # init_time is used naked so we must strip it here to match.
        return {"date": init_date + dt.timedelta(hours=6), "fields": fields}

    def _init_naive(self, init_date):
        return init_date.replace(tzinfo=None)

    def test_basic_schema_drops_pressure_vars(
        self, init_date, identity_regridder, aifs_fields
    ):
        ds = _app.state_to_xarray(
            self._state(init_date, aifs_fields),
            regridder=identity_regridder,
            init_time=self._init_naive(init_date),
        )
        for d in ("init_time", "lead_time", "lat", "lon"):
            assert d in ds.dims
        assert "10u" in ds.data_vars
        assert "q" not in ds.data_vars
        assert "q_50" not in ds.data_vars

    def test_include_pressure_levels_builds_composite(
        self, init_date, identity_regridder, aifs_fields
    ):
        ds = _app.state_to_xarray(
            self._state(init_date, aifs_fields),
            regridder=identity_regridder,
            init_time=self._init_naive(init_date),
            include_pressure_levels=True,
        )
        for p in _app.PRESSURE_VAR_PREFIXES:
            assert p in ds.data_vars
            assert "pressure" in ds[p].dims
        assert "q_50" not in ds.data_vars

    def test_lead_time_is_difference(self, init_date, identity_regridder, aifs_fields):
        ds = _app.state_to_xarray(
            self._state(init_date, aifs_fields),
            regridder=identity_regridder,
            init_time=self._init_naive(init_date),
        )
        assert ds["lead_time"].values[0] == np.timedelta64(6, "h")


class TestRunMember:
    def test_drops_extra_vars_and_concats_lead_times(
        self, init_date, fake_runner, identity_regridder, aifs_fields
    ):
        extras = {"not_a_real_var": np.zeros(32, dtype="f4")}
        ds = _app._run_member(
            init_date,
            fake_runner,
            {**aifs_fields, **extras},
            identity_regridder,
            lead_time=12,
        )
        assert ds.sizes["lead_time"] == 2
        assert "not_a_real_var" not in ds.data_vars
        assert "ensemble_member" not in ds.dims

    def test_member_id_expands_ensemble_dim(
        self, init_date, fake_runner, identity_regridder, aifs_fields
    ):
        ds = _app._run_member(
            init_date,
            fake_runner,
            aifs_fields,
            identity_regridder,
            member_id=3,
            lead_time=6,
        )
        assert ds["ensemble_member"].values.tolist() == [3]


# ===========================================================================
# _ic.py — initial-conditions store
# ===========================================================================
class TestParseUtcDate:
    @pytest.mark.parametrize("hour", [0, 6, 12, 18])
    def test_valid_hours(self, hour):
        out = _ic._parse_utc_date(f"2024-01-01T{hour:02d}:00:00")
        assert out.tzinfo == dt.UTC and out.hour == hour

    def test_naive_string_becomes_utc(self):
        assert _ic._parse_utc_date("2024-01-01T00:00:00").tzinfo == dt.UTC

    def test_rejects_minutes(self):
        with pytest.raises(ValueError, match="hourly"):
            _ic._parse_utc_date("2024-01-01T00:30:00")

    def test_rejects_off_hour(self):
        with pytest.raises(ValueError, match="00, 06, 12 or 18"):
            _ic._parse_utc_date("2024-01-01T03:00:00")


class TestIterDates6h:
    def test_yields_inclusive_endpoints(self):
        start = dt.datetime(2024, 1, 1, 0, tzinfo=dt.UTC)
        end = dt.datetime(2024, 1, 1, 18, tzinfo=dt.UTC)
        out = list(_ic._iter_dates_6h(start, end))
        assert len(out) == 4
        assert out[0] == start and out[-1] == end


class TestIcFileRoundtrip:
    def test_stack_fields_shape(self):
        names, stacked = _ic._stack_fields({"a": np.arange(5), "b": np.arange(5) + 1})
        assert names == ["a", "b"]
        assert stacked.shape == (2, 5)

    def test_store_and_fetch_roundtrip(self, ic_dir, init_date, write_ic_dates):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        fields = _ic.fetch_initial_conditions(init_date, ic_dir)
        assert "tcw" in fields and "tcwv" not in fields
        assert "stl1" in fields and "sot_1" not in fields
        assert "swvl1" in fields and "vsw_1" not in fields
        for lvl in _ic.LEVELS:
            assert f"z_{lvl}" in fields and f"gh_{lvl}" not in fields
        assert fields["10u"].shape == (2, 8)

    def test_delete_ic_dates_removes_dirs(self, ic_dir, init_date, write_ic_dates):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        _ic.delete_ic_dates(init_date, ic_dir)
        for d in [init_date - dt.timedelta(hours=6), init_date]:
            assert not os.path.exists(os.path.join(ic_dir, utils.datetime_to_str(d)))


class TestIngestRange:
    def test_skips_existing_writes_missing(self, ic_dir, init_date, rng):
        start = init_date
        end = init_date + dt.timedelta(hours=6)
        data = {"x": rng.random(5, dtype="f4")}
        _ic.get_and_store_date(start, ic_dir, lambda _d, _data=data: _data)

        called_for = []

        def fetch_fn(d):
            called_for.append(d)
            return {"x": rng.random(5, dtype="f4")}

        _ic._ingest_range(
            start.isoformat(), end.isoformat(), ic_dir, fetch_fn, source="test"
        )
        assert called_for == [end]

    def test_rejects_end_before_start(self, ic_dir):
        with pytest.raises(ValueError, match="end_date must be"):
            _ic._ingest_range(
                "2024-01-02T00:00:00",
                "2024-01-01T00:00:00",
                ic_dir,
                lambda d: {"x": np.zeros(3, "f4")},
                source="test",
            )

    def test_noop_when_all_present(self, ic_dir, init_date, rng, capsys):
        # only one date in range, pre-populate it
        data = {"x": rng.random(5, dtype="f4")}
        _ic.get_and_store_date(init_date, ic_dir, lambda _d, _data=data: _data)

        def fetch_fn(d):
            raise AssertionError("should not be called")

        _ic._ingest_range(
            init_date.isoformat(),
            init_date.isoformat(),
            ic_dir,
            fetch_fn,
            source="test",
        )
        assert "already present" in capsys.readouterr().out

    def test_reingests_incomplete_date_dir(self, ic_dir, init_date, rng):
        # empty dir from a crashed ingest is rewritten, not skipped
        date_path = os.path.join(ic_dir, utils.datetime_to_str(init_date))
        os.makedirs(date_path)

        called = []

        def fetch_fn(d):
            called.append(d)
            return {"x": rng.random(5, dtype="f4")}

        _ic._ingest_range(
            init_date.isoformat(),
            init_date.isoformat(),
            ic_dir,
            fetch_fn,
            source="test",
        )
        assert called == [init_date]
        assert _ic.ic_date_complete(date_path)


class TestIcDateComplete:
    def test_true_after_store(self, ic_dir, init_date, rng):
        _ic.get_and_store_date(
            init_date, ic_dir, lambda d: {"x": rng.random(5, dtype="f4")}
        )
        assert _ic.ic_date_complete(
            os.path.join(ic_dir, utils.datetime_to_str(init_date))
        )

    def test_false_for_empty_dir(self, tmp_path):
        assert not _ic.ic_date_complete(str(tmp_path))

    def test_false_for_missing_dir(self, tmp_path):
        assert not _ic.ic_date_complete(str(tmp_path / "nope"))

    def test_failed_fetch_leaves_no_dir(self, ic_dir, init_date):
        # group must only be created once fetch/regrid has produced data
        def boom(d):
            raise RuntimeError("regrid crashed")

        with pytest.raises(RuntimeError, match="regrid crashed"):
            _ic.get_and_store_date(init_date, ic_dir, boom)
        assert not os.path.exists(
            os.path.join(ic_dir, utils.datetime_to_str(init_date))
        )


# ===========================================================================
# _ingest_ekd.py
# ===========================================================================
class _EkdField:
    """Minimal stand-in for an earthkit-data field."""

    def __init__(self, **meta):
        self._meta = meta

    def metadata(self, key):
        return self._meta[key]


class TestIngestEkd:
    def test_field_to_n320_rolls_negative_longitudes(self, monkeypatch, rng):
        captured = {}

        def fake_regrid(arr):
            captured["arr"] = arr.copy()
            return np.array([1.0], dtype="f4")

        monkeypatch.setattr(_ingest_ekd, "_regrid_n320", fake_regrid)
        arr = rng.random((721, 1440), dtype="f4")

        class _Field:
            def to_numpy(self, dtype):
                return arr.astype(dtype)

            def metadata(self, key):
                return -180.0  # triggers roll

        _ingest_ekd._field_to_n320(_Field())
        np.testing.assert_array_equal(captured["arr"], np.roll(arr, -720, axis=1))

    def test_field_to_n320_no_roll_for_positive_lon0(self, monkeypatch, rng):
        captured = {}

        def fake_regrid(arr):
            captured["arr"] = arr.copy()
            return np.array([1.0], dtype="f4")

        monkeypatch.setattr(_ingest_ekd, "_regrid_n320", fake_regrid)
        arr = rng.random((721, 1440), dtype="f4")

        class _Field:
            def to_numpy(self, dtype):
                return arr.astype(dtype)

            def metadata(self, key):
                return 0.0

        _ingest_ekd._field_to_n320(_Field())
        np.testing.assert_array_equal(captured["arr"], arr)

    def test_cds_request_schema(self):
        d = dt.datetime(2024, 3, 15, 12, tzinfo=dt.UTC)
        r = _ingest_ekd._cds_request(d, variable=["x"], pressure_level=["50"])
        assert r["product_type"] == "reanalysis"
        assert r["grid"] == [0.25, 0.25]
        assert r["date"] == "2024-03-15"
        assert r["time"] == "12:00"
        assert r["variable"] == ["x"]
        assert r["pressure_level"] == ["50"]

    def test_ingest_dispatches_to_open_data(self, monkeypatch, ic_dir):
        seen = {}

        def fake_ingest_range(s, e, d, fetch_fn, *, source):
            seen["source"] = source
            seen["fetch_fn"] = fetch_fn

        monkeypatch.setattr(_ic, "_ingest_range", fake_ingest_range)
        _ingest_ekd.ingest("2024-01-01", "2024-01-01", ic_dir, "ifs-ekd")
        assert seen["source"] == "ifs-ekd"
        assert seen["fetch_fn"] is _ingest_ekd._get_all_open_data

    def test_ingest_dispatches_to_cds(self, monkeypatch, ic_dir):
        seen = {}

        def fake_ingest_range(s, e, d, fetch_fn, *, source):
            seen["source"] = source
            seen["fetch_fn"] = fetch_fn

        monkeypatch.setattr(_ic, "_ingest_range", fake_ingest_range)
        _ingest_ekd.ingest("2024-01-01", "2024-01-01", ic_dir, "era5-cds")
        assert seen["source"] == "era5-cds"
        assert seen["fetch_fn"] is _ingest_ekd._get_all_cds

    def test_get_open_data_raises_on_missing(self, monkeypatch):
        def boom(*a, **kw):
            raise FileNotFoundError("nope")

        monkeypatch.setattr(_ingest_ekd.ekd, "from_source", boom)
        with pytest.raises(RuntimeError, match="Failed to fetch ecmwf-open-data"):
            _ingest_ekd._get_open_data(
                dt.datetime(2024, 1, 1, tzinfo=dt.UTC), "10u", []
            )

    def test_get_open_data_field_naming(self, monkeypatch):
        monkeypatch.setattr(
            _ingest_ekd, "_field_to_n320", lambda f: np.ones(3, dtype="f4")
        )
        seen = {}

        def fake_from_source(kind, **kwargs):
            seen["kind"] = kind
            seen["kwargs"] = kwargs
            return [_EkdField(param="t", levelist=500)]

        monkeypatch.setattr(_ingest_ekd.ekd, "from_source", fake_from_source)
        date = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)

        # with levelist: keys are "<param>_<level>"
        out = _ingest_ekd._get_open_data(date, "t", [500])
        assert set(out) == {"t_500"}
        assert seen["kind"] == "ecmwf-open-data"
        assert seen["kwargs"]["date"].tzinfo is None  # date made naive
        assert seen["kwargs"]["source"] == "aws"

        # default levelist=None → empty list and plain param key
        out = _ingest_ekd._get_open_data(date, "t")
        assert set(out) == {"t"}
        assert seen["kwargs"]["levelist"] == []

    def test_get_all_open_data_queries_every_param(self, monkeypatch):
        calls = []

        def fake_get(date, param, levelist=None):
            calls.append((param, levelist))
            key = param if levelist is None else f"{param}_{levelist[0]}"
            return {key: np.ones(3, dtype="f4")}

        monkeypatch.setattr(_ingest_ekd, "_get_open_data", fake_get)
        out = _ingest_ekd._get_all_open_data(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))

        assert [c[0] for c in calls] == (
            _ingest_ekd.PARAM_SFC + _ingest_ekd.PARAM_SOIL + _ingest_ekd.PARAM_PL
        )
        for param, levelist in calls:
            if param in _ingest_ekd.PARAM_SOIL:
                assert levelist == _ingest_ekd.SOIL_LEVELS
            elif param in _ingest_ekd.PARAM_PL:
                assert levelist == _ingest_ekd.LEVELS
            else:
                assert levelist is None
        assert len(out) == len(calls)

    def test_get_cds_sfc_keys_by_shortname(self, monkeypatch):
        monkeypatch.setattr(
            _ingest_ekd, "_field_to_n320", lambda f: np.ones(3, dtype="f4")
        )
        monkeypatch.setattr(
            _ingest_ekd.ekd,
            "from_source",
            lambda *a, **kw: [_EkdField(shortName="2t"), _EkdField(shortName="msl")],
        )
        out = _ingest_ekd._get_cds_sfc(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
        assert set(out) == {"2t", "msl"}

    def test_get_cds_soil_maps_to_storage_keys(self, monkeypatch):
        monkeypatch.setattr(
            _ingest_ekd, "_field_to_n320", lambda f: np.ones(3, dtype="f4")
        )
        monkeypatch.setattr(
            _ingest_ekd.ekd,
            "from_source",
            lambda *a, **kw: [
                _EkdField(shortName="swvl1"),
                _EkdField(shortName="stl2"),
            ],
        )
        out = _ingest_ekd._get_cds_soil(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
        assert set(out) == {"vsw_1", "sot_2"}

    def test_get_cds_pl_converts_z_to_gh(self, monkeypatch):
        monkeypatch.setattr(
            _ingest_ekd,
            "_field_to_n320",
            lambda f: np.full(3, 2 * _ic._G, dtype="f4"),
        )
        monkeypatch.setattr(
            _ingest_ekd.ekd,
            "from_source",
            lambda *a, **kw: [
                _EkdField(shortName="z", level=500),
                _EkdField(shortName="t", level=850),
            ],
        )
        out = _ingest_ekd._get_cds_pl(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
        assert set(out) == {"gh_500", "t_850"}
        np.testing.assert_allclose(out["gh_500"], 2.0)
        np.testing.assert_allclose(out["t_850"], 2 * _ic._G)

    def test_get_all_cds_merges(self, monkeypatch):
        monkeypatch.setattr(_ingest_ekd, "_get_cds_sfc", lambda d: {"a": 1})
        monkeypatch.setattr(_ingest_ekd, "_get_cds_soil", lambda d: {"b": 2})
        monkeypatch.setattr(_ingest_ekd, "_get_cds_pl", lambda d: {"c": 3})
        out = _ingest_ekd._get_all_cds(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
        assert out == {"a": 1, "b": 2, "c": 3}


# ===========================================================================
# _ingest_era5_arco.py
# ===========================================================================
class TestIngestArco:
    def test_get_time_index_known_date(self):
        # lru_cache(maxsize=1) means only one result at a time — clear first.
        _ingest_era5_arco._get_time_index.cache_clear()
        # 5 hours past the 1900-01-01 epoch == 5
        assert _ingest_era5_arco._get_time_index(dt.datetime(1900, 1, 1, 5)) == 5

    def test_get_array_dispatch_by_shape(self, rng):
        level_arr = np.array([50, 500, 1000], dtype="i4")
        arr2 = rng.random((721, 1440), dtype="f4")
        arr3 = rng.random((10, 721, 1440), dtype="f4")
        arr4 = rng.random((10, 3, 721, 1440), dtype="f4")

        class _Group:
            def __getitem__(self, k):
                return {
                    "level": level_arr,
                    "two_d": arr2,
                    "three_d": arr3,
                    "four_d": arr4,
                }[k]

        g = _Group()
        np.testing.assert_array_equal(
            _ingest_era5_arco._get_array(g, "two_d", time_index=5), arr2
        )
        np.testing.assert_array_equal(
            _ingest_era5_arco._get_array(g, "three_d", time_index=5), arr3[5]
        )
        np.testing.assert_array_equal(
            _ingest_era5_arco._get_array(g, "four_d", time_index=5, level=500),
            arr4[5, 1],
        )
        with pytest.raises(ValueError, match="level required"):
            _ingest_era5_arco._get_array(g, "four_d", time_index=5)

    def test_get_array_3d_with_level_indexes_level(self, rng):
        # 3D array shaped (level, lat, lon) — level given, no time dimension
        level_arr = np.array([50, 500, 1000], dtype="i4")
        arr3 = rng.random((3, 7, 9), dtype="f4")

        class _Group:
            def __getitem__(self, k):
                return {"level": level_arr, "x": arr3}[k]

        np.testing.assert_array_equal(
            _ingest_era5_arco._get_array(_Group(), "x", time_index=2, level=1000),
            arr3[2, 2],
        )

    def test_get_array_unexpected_shape_raises(self, rng):
        class _Group:
            def __getitem__(self, k):
                return np.zeros(5, dtype="f4")

        with pytest.raises(ValueError, match="Unexpected array shape"):
            _ingest_era5_arco._get_array(_Group(), "one_d", time_index=0)

    def test_get_all_data_fetches_full_keyset(self, monkeypatch):
        # the ARCO store's level coordinate is ascending (searchsorted relies on it)
        levels = np.sort(np.array(_ingest_era5_arco.LEVELS, dtype="i4"))
        n_time, n_lat, n_lon = 2, 3, 4
        consts = {}
        arrays = {"level": levels}
        sfc_names = {**_ingest_era5_arco._SFC_ARCO, **_ingest_era5_arco._SOIL_ARCO}
        for i, arco_name in enumerate(sfc_names.values()):
            consts[arco_name] = float(i + 1)
            arrays[arco_name] = np.full(
                (n_time, n_lat, n_lon), consts[arco_name], dtype="f4"
            )
        for j, arco_name in enumerate(_ingest_era5_arco._PL_ARCO.values()):
            consts[arco_name] = float(100 + j)
            arrays[arco_name] = np.full(
                (n_time, len(levels), n_lat, n_lon), consts[arco_name], dtype="f4"
            )

        class _Group:
            def __getitem__(self, k):
                return arrays[k]

        monkeypatch.setattr(_ingest_era5_arco, "_open_arco", lambda: _Group())
        monkeypatch.setattr(_ingest_era5_arco, "_regrid_n320", lambda a: a)
        _ingest_era5_arco._get_time_index.cache_clear()

        # 1900-01-01 01:00 → time index 1 (< n_time)
        data = _ingest_era5_arco.get_all_data(dt.datetime(1900, 1, 1, 1, tzinfo=dt.UTC))

        expected = (
            set(_ingest_era5_arco._SFC_ARCO)
            | set(_ingest_era5_arco._SOIL_ARCO)
            | {
                f"{p}_{lvl}"
                for p in _ingest_era5_arco._PL_ARCO
                for lvl in _ingest_era5_arco.LEVELS
            }
        )
        assert set(data) == expected
        np.testing.assert_allclose(data["2t"], consts["2m_temperature"])
        np.testing.assert_allclose(
            data["vsw_1"], consts["volumetric_soil_water_layer_1"]
        )
        np.testing.assert_allclose(data["t_500"], consts["temperature"])
        # geopotential is converted to geopotential height
        np.testing.assert_allclose(
            data["gh_500"], consts["geopotential"] / _ingest_era5_arco._G
        )

    def test_ingest_dispatches_to_ingest_range(self, monkeypatch, ic_dir):
        seen = {}

        def fake_ingest_range(s, e, d, fetch_fn, *, source):
            seen["fetch_fn"] = fetch_fn
            seen["source"] = source

        monkeypatch.setattr(_ic, "_ingest_range", fake_ingest_range)
        _ingest_era5_arco.ingest("2024-01-01", "2024-01-01", ic_dir)
        assert seen["fetch_fn"] is _ingest_era5_arco.get_all_data
        assert seen["source"] == "era5-arco"

    def test_open_arco_opens_anonymous_readonly(self, monkeypatch):
        fs_kwargs = {}
        monkeypatch.setattr(
            _ingest_era5_arco.gcsfs,
            "GCSFileSystem",
            lambda **kw: fs_kwargs.update(kw) or "FS",
        )
        monkeypatch.setattr(
            _ingest_era5_arco.zarr.storage,
            "FsspecStore",
            lambda fs, path: ("STORE", fs, path),
        )
        monkeypatch.setattr(
            _ingest_era5_arco.zarr,
            "open",
            lambda store, mode: ("GROUP", store, mode),
        )
        _ingest_era5_arco._open_arco.cache_clear()
        try:
            group = _ingest_era5_arco._open_arco()
        finally:
            # don't leave the fake group in the lru_cache for other tests
            _ingest_era5_arco._open_arco.cache_clear()
        assert fs_kwargs["token"] == "anon"
        assert fs_kwargs["access"] == "read_only"
        assert group == (
            "GROUP",
            ("STORE", "FS", _ingest_era5_arco._ARCO_PATH),
            "r",
        )


# ===========================================================================
# _ingest_ifs_arraylake.py — pure-data logic
# ===========================================================================
def _bb_dataset(rng, n_init=2, n_lat=4, n_lon=8, levels=None):
    levels = levels if levels is not None else _ic.LEVELS
    init_times = np.array(
        [dt.datetime(2024, 1, 1, 0) + dt.timedelta(hours=6 * i) for i in range(n_init)],
        dtype="datetime64[ns]",
    )
    sfc_vars = ["u10", "v10", "d2m", "t2m", "msl", "skt", "sp", "tcwv"]
    pl_vars = ["u", "v", "t", "q", "w", "z"]
    data = {
        v: (
            ("init_time", "lead_time", "latitude", "longitude"),
            rng.random((n_init, 1, n_lat, n_lon), dtype="f4"),
        )
        for v in sfc_vars
    }
    for v in pl_vars:
        data[v] = (
            ("init_time", "lead_time", "level", "latitude", "longitude"),
            rng.random((n_init, 1, len(levels), n_lat, n_lon), dtype="f4"),
        )
    return xr.Dataset(
        data,
        coords={
            "init_time": init_times,
            "lead_time": np.array([0], dtype="timedelta64[ns]"),
            "level": np.array(levels, dtype="i4"),
            "latitude": np.linspace(90, -90, n_lat),
            "longitude": np.linspace(0, 359.75, n_lon),
        },
    )


class TestIngestArraylake:
    def test_build_rename_map_picks_tcwv_over_tcw(self, rng):
        ds = _bb_dataset(rng)
        ds = ds.assign(tcw=ds["tcwv"].copy())
        m = _ingest_ifs_arraylake._build_rename_map(ds)
        assert m.get("tcwv") == "tcwv"
        assert "tcw" not in m

    def test_build_rename_map_includes_soil(self, rng):
        ds = _bb_dataset(rng)
        ds = ds.assign(stl1=ds["t2m"].copy(), swvl2=ds["t2m"].copy())
        m = _ingest_ifs_arraylake._build_rename_map(ds)
        assert m["stl1"] == "sot_1"
        assert m["swvl2"] == "vsw_2"

    def test_to_eastward_normalizes_negative_longitudes(self):
        ds = xr.Dataset(
            {"v": (("longitude",), np.array([0.0, 1.0, 2.0, 3.0], dtype="f4"))},
            coords={"longitude": np.array([-180.0, -90.0, 0.0, 90.0])},
        )
        out = _ingest_ifs_arraylake._to_eastward_0_360(ds)
        np.testing.assert_array_equal(out["longitude"].values, [0, 90, 180, 270])
        np.testing.assert_array_equal(out["v"].values, [2, 3, 0, 1])

    def test_to_eastward_noop_when_already_0_360(self):
        ds = xr.Dataset(
            {"v": (("longitude",), np.arange(4, dtype="f4"))},
            coords={"longitude": np.array([0.0, 90.0, 180.0, 270.0])},
        )
        assert _ingest_ifs_arraylake._to_eastward_0_360(ds) is ds

    def test_to_eastward_noop_without_longitude_coord(self):
        ds = xr.Dataset({"v": (("x",), np.arange(4, dtype="f4"))})
        assert _ingest_ifs_arraylake._to_eastward_0_360(ds) is ds

    def test_pl_src_prefixes(self, rng):
        ds = _bb_dataset(rng)
        assert set(_ingest_ifs_arraylake._pl_src_prefixes(ds)) == {
            "u",
            "v",
            "t",
            "q",
            "w",
            "z",
        }

    def test_read_static_fields(self, monkeypatch, rng):
        ds = xr.Dataset(
            {
                "lsm": (("y", "x"), rng.random((4, 8), dtype="f4")),
                "z_sfc": (("y", "x"), rng.random((4, 8), dtype="f4")),
                "slor": (("y", "x"), rng.random((4, 8), dtype="f4")),
                "sdor": (("y", "x"), rng.random((4, 8), dtype="f4")),
            }
        )
        monkeypatch.setattr(
            _ingest_ifs_arraylake,
            "_regrid_n320",
            lambda a: a.ravel()[:3].astype("f4"),
        )
        out = _ingest_ifs_arraylake._read_static_fields(ds)
        assert set(out) == {"lsm", "z", "slor", "sdor"}
        for v in out.values():
            assert v.shape == (3,) and v.dtype == np.dtype("f4")

    def test_read_static_fields_squeezes_leading_dims(self, monkeypatch, rng):
        ds = xr.Dataset(
            {
                src: (("time", "y", "x"), rng.random((1, 4, 8), dtype="f4"))
                for src in _ingest_ifs_arraylake._STATIC_VARS
            }
        )
        seen_ndims = []

        def fake_regrid(a):
            seen_ndims.append(a.ndim)
            return a.ravel()[:3].astype("f4")

        monkeypatch.setattr(_ingest_ifs_arraylake, "_regrid_n320", fake_regrid)
        out = _ingest_ifs_arraylake._read_static_fields(ds)
        assert set(out) == {"lsm", "z", "slor", "sdor"}
        assert seen_ndims == [2] * 4

    def test_ingest_writes_missing_dates(self, monkeypatch, rng, ic_dir):
        ds = _bb_dataset(rng)
        monkeypatch.setattr(
            _ingest_ifs_arraylake,
            "_regrid_n320",
            lambda a: a.ravel()[:3].astype("f4"),
        )
        _ingest_ifs_arraylake.ingest(
            "2024-01-01T00:00:00",
            "2024-01-01T06:00:00",
            ic_dir,
            ds,
            {"lsm": np.zeros(3, "f4")},
        )
        for d in [
            dt.datetime(2024, 1, 1, 0, tzinfo=dt.UTC),
            dt.datetime(2024, 1, 1, 6, tzinfo=dt.UTC),
        ]:
            assert os.path.isdir(os.path.join(ic_dir, utils.datetime_to_str(d)))

    def test_ingest_skips_when_all_present(self, monkeypatch, rng, ic_dir, capsys):
        ds = _bb_dataset(rng)
        monkeypatch.setattr(
            _ingest_ifs_arraylake,
            "_regrid_n320",
            lambda a: a.ravel()[:3].astype("f4"),
        )
        for d in [
            dt.datetime(2024, 1, 1, 0, tzinfo=dt.UTC),
            dt.datetime(2024, 1, 1, 6, tzinfo=dt.UTC),
        ]:
            os.makedirs(os.path.join(ic_dir, utils.datetime_to_str(d)))

        _ingest_ifs_arraylake.ingest(
            "2024-01-01T00:00:00",
            "2024-01-01T06:00:00",
            ic_dir,
            ds,
            {"lsm": np.zeros(3, "f4")},
        )
        assert "already present" in capsys.readouterr().out

    def test_ingest_rejects_end_before_start(self, rng, ic_dir):
        ds = _bb_dataset(rng)
        with pytest.raises(ValueError, match="end_date must be"):
            _ingest_ifs_arraylake.ingest(
                "2024-01-02T00:00:00",
                "2024-01-01T00:00:00",
                ic_dir,
                ds,
                {},
            )


# ===========================================================================
# utils.py
# ===========================================================================
class TestDatetimeToStr:
    def test_format(self, init_date):
        assert utils.datetime_to_str(init_date) == "2024-01-01/00z"

    def test_rejects_naive(self):
        with pytest.raises(AssertionError):
            utils.datetime_to_str(dt.datetime(2024, 1, 1, 0))


class TestApplyTrajectoryChunks:
    def test_sets_encoding(self, rng):
        ds = xr.Dataset(
            {
                "t": (
                    ("init_time", "lead_time", "lat", "lon"),
                    rng.random((2, 4, 50, 60)),
                )
            }
        )
        utils.apply_trajectory_chunks(ds)
        # init_time clipped to 1, others remain (since 50/60 < 241/240)
        assert ds["t"].encoding["chunks"] == (1, 4, 50, 60)


class TestGetStorage:
    @pytest.fixture(autouse=True)
    def _aws_creds(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
        monkeypatch.setenv("AWS_REGION", "auto")
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "acc")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown storage_type"):
            utils.get_storage("b", "p", storage_type="dropbox")

    @pytest.mark.parametrize("kind", ["tigris", "s3", "gcs"])
    def test_known_returns_storage(self, kind):
        # tigris/s3/gcs construct cleanly with from_env or explicit credentials
        s = utils.get_storage("bucket", "prefix", storage_type=kind)
        assert isinstance(s, icechunk.Storage)

    @pytest.mark.parametrize("kind", ["r2", "azure"])
    def test_dispatch_to_constructor(self, monkeypatch, kind):
        # r2 / azure need extra config (account_id, etc.) at construction time.
        # Patch icechunk.<kind>_storage to a sentinel so we can prove the
        # dispatch branch is hit without supplying real credentials.
        called = {}

        def _fake(**kwargs):
            called.update(kwargs)
            return f"FAKE-{kind}"

        monkeypatch.setattr(icechunk, f"{kind}_storage", _fake)
        out = utils.get_storage("bucket", "prefix", storage_type=kind)
        assert out == f"FAKE-{kind}"
        assert called["prefix"] == "prefix"


class TestForecastExists:
    def _patch_to_local(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            utils,
            "get_storage",
            lambda *a, **kw: icechunk.local_filesystem_storage(
                str(tmp_path / "outputs")
            ),
        )

    def test_false_when_group_missing(self, monkeypatch, tmp_path, init_date):
        self._patch_to_local(monkeypatch, tmp_path)
        # creates an empty repo with default main branch but no group
        assert (
            utils.forecast_exists(init_date, "bucket", outputs_prefix="run1") is False
        )

    def test_true_after_write(self, tmp_path, init_date, rng, monkeypatch):
        self._patch_to_local(monkeypatch, tmp_path)
        repo = icechunk.Repository.open_or_create(
            icechunk.local_filesystem_storage(str(tmp_path / "outputs"))
        )
        sess = repo.writable_session("main")
        xr.Dataset({"x": (("a",), rng.random(3))}).to_zarr(
            sess.store,
            group=utils.datetime_to_str(init_date),
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
        sess.commit("seed")
        assert utils.forecast_exists(init_date, "bucket", outputs_prefix="any") is True

    def test_n_members_threshold(self, tmp_path, init_date, rng, monkeypatch):
        self._patch_to_local(monkeypatch, tmp_path)
        repo = icechunk.Repository.open_or_create(
            icechunk.local_filesystem_storage(str(tmp_path / "outputs"))
        )
        sess = repo.writable_session("main")
        xr.Dataset(
            {"x": (("ensemble_member", "a"), rng.random((2, 3)))},
            coords={"ensemble_member": [0, 1]},
        ).to_zarr(
            sess.store,
            group=utils.datetime_to_str(init_date),
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
        sess.commit("seed")
        assert (
            utils.forecast_exists(
                init_date, "bucket", outputs_prefix="any", n_members=5
            )
            is False
        )
        assert (
            utils.forecast_exists(
                init_date, "bucket", outputs_prefix="any", n_members=2
            )
            is True
        )

    def test_false_when_branch_missing(self, monkeypatch, tmp_path, init_date):
        self._patch_to_local(monkeypatch, tmp_path)
        # Pre-create repo, then call forecast_exists on a non-existent branch
        icechunk.Repository.open_or_create(
            icechunk.local_filesystem_storage(str(tmp_path / "outputs"))
        )
        assert (
            utils.forecast_exists(
                init_date,
                "bucket",
                outputs_prefix="any",
                outputs_branch="does-not-exist",
            )
            is False
        )


# ===========================================================================
# _run_forecast_impl — orchestration dispatch
# ===========================================================================
class _Recorder:
    """Callable that records its positional + keyword arguments."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.fixture
def recorders():
    return {
        "ingest_era5_arco_fn": _Recorder(),
        "ingest_ifs_arraylake_fn": _Recorder(),
        "ingest_ekd_fn": _Recorder(),
        "run_inference_fn": _Recorder(),
    }


def _call_forecast(recorders, **overrides):
    """Call ``_run_forecast_impl`` with safe defaults + overrides."""
    kwargs = dict(
        date=dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC),
        storage_bucket="bucket",
        ic_dir="/nonexistent",
        outputs_prefix="run1",
        open_outputs_repo_fn=mock.Mock(side_effect=RuntimeError("no repo")),
        **recorders,
    )
    kwargs.update(overrides)
    _app._run_forecast_impl(**kwargs)


class TestRunForecastImpl:
    def test_requires_outputs_target(self, recorders):
        with pytest.raises(ValueError, match="outputs_prefix"):
            _call_forecast(recorders, outputs_prefix=None, outputs_repo=None)

    def test_dispatches_era5_arco(self, recorders):
        _call_forecast(recorders, ic_source="era5-arco")
        assert len(recorders["ingest_era5_arco_fn"].calls) == 1
        assert recorders["ingest_era5_arco_fn"].calls[0][0] == (
            "2024-01-01T06:00:00+00:00",
            "2024-01-01T12:00:00+00:00",
        )
        assert len(recorders["run_inference_fn"].calls) == 1

    def test_dispatches_ifs_arraylake(self, recorders):
        _call_forecast(
            recorders,
            ic_source="ifs-arraylake",
            ic_source_repo="org/al",
            ic_source_branch="dev",
        )
        args, kwargs = recorders["ingest_ifs_arraylake_fn"].calls[0]
        assert args[2] == "org/al"
        assert kwargs["ic_source_branch"] == "dev"

    def test_ifs_arraylake_without_repo_raises(self, recorders):
        with pytest.raises(ValueError, match="ic_source_repo is required"):
            _call_forecast(recorders, ic_source="ifs-arraylake")

    def test_dispatches_ifs_ekd(self, recorders):
        _call_forecast(recorders, ic_source="ifs-ekd")
        assert len(recorders["ingest_ekd_fn"].calls) == 1
        assert recorders["ingest_ekd_fn"].calls[0][0][3] == "ifs-ekd"

    def test_skips_ingest_when_ics_present(
        self, recorders, ic_dir, init_date, write_ic_dates
    ):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        _call_forecast(
            recorders,
            date=init_date,
            ic_dir=ic_dir,
            ic_source="ifs-ekd",
        )
        assert recorders["ingest_ekd_fn"].calls == []
        assert len(recorders["run_inference_fn"].calls) == 1

    def test_reingests_when_ic_dirs_incomplete(self, recorders, tmp_path, init_date):
        # empty date dirs from a crashed ingest must trigger re-ingestion
        for d in [init_date - dt.timedelta(hours=6), init_date]:
            (tmp_path / utils.datetime_to_str(d)).mkdir(parents=True)
        _call_forecast(
            recorders,
            date=init_date,
            ic_dir=str(tmp_path),
            ic_source="ifs-ekd",
        )
        assert len(recorders["ingest_ekd_fn"].calls) == 1

    def test_skips_when_existing_forecast(
        self, recorders, init_date, local_outputs_repo, rng
    ):
        sess = local_outputs_repo.writable_session("main")
        xr.Dataset({"x": (("a",), rng.random(3))}).to_zarr(
            sess.store,
            group=utils.datetime_to_str(init_date),
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
        sess.commit("seed")

        _call_forecast(
            recorders,
            date=init_date,
            open_outputs_repo_fn=lambda *a, **kw: local_outputs_repo,
        )
        assert recorders["run_inference_fn"].calls == []
        assert recorders["ingest_ekd_fn"].calls == []

    def test_default_checkpoint_single(self, recorders):
        _call_forecast(recorders, ic_source="ifs-ekd")
        ckpt = recorders["run_inference_fn"].calls[0][0][2]
        assert ckpt == settings.AIFS_SINGLE_CHECKPOINT

    def test_default_checkpoint_ens(self, recorders):
        _call_forecast(recorders, ic_source="ifs-ekd", n_members=4)
        ckpt = recorders["run_inference_fn"].calls[0][0][2]
        assert ckpt == settings.AIFS_ENS_CHECKPOINT

    def test_explicit_checkpoint_passthrough(self, recorders):
        ckpt = {"huggingface": "custom/model"}
        _call_forecast(recorders, ic_source="ifs-ekd", checkpoint=ckpt)
        assert recorders["run_inference_fn"].calls[0][0][2] == ckpt

    def test_on_ic_reload_called(self, recorders):
        reload_mock = mock.Mock()
        _call_forecast(recorders, ic_source="ifs-ekd", on_ic_reload=reload_mock)
        reload_mock.assert_called_once()

    def test_overwrite_bypasses_skip_check(
        self, recorders, init_date, local_outputs_repo, rng
    ):
        sess = local_outputs_repo.writable_session("main")
        xr.Dataset({"x": (("a",), rng.random(3))}).to_zarr(
            sess.store,
            group=utils.datetime_to_str(init_date),
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
        sess.commit("seed")
        _call_forecast(
            recorders,
            date=init_date,
            open_outputs_repo_fn=lambda *a, **kw: local_outputs_repo,
            overwrite=True,
            ic_source="ifs-ekd",
        )
        assert len(recorders["run_inference_fn"].calls) == 1

    def test_default_ic_source_used(self, recorders):
        # default is "ifs-arraylake" → must error without ic_source_repo
        with pytest.raises(ValueError, match="ic_source_repo is required"):
            _call_forecast(recorders)

    def test_missing_outputs_branch_does_not_skip(
        self, recorders, init_date, local_outputs_repo, rng
    ):
        # forecast exists on main, but the requested branch doesn't exist →
        # the skip check must not trigger and inference must be dispatched
        sess = local_outputs_repo.writable_session("main")
        xr.Dataset({"x": (("a",), rng.random(3))}).to_zarr(
            sess.store,
            group=utils.datetime_to_str(init_date),
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
        sess.commit("seed")
        _call_forecast(
            recorders,
            date=init_date,
            open_outputs_repo_fn=lambda *a, **kw: local_outputs_repo,
            outputs_branch="not-there",
            ic_source="ifs-ekd",
        )
        assert len(recorders["run_inference_fn"].calls) == 1


# ===========================================================================
# _run_inference_impl — local-FS icechunk + fake runner
# ===========================================================================
class TestRunInferenceImpl:
    def test_writes_forecast_and_deletes_ics(
        self,
        ic_dir,
        local_outputs_repo,
        write_ic_dates,
        fake_runner,
        identity_regridder,
        init_date,
    ):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        ran = _app._run_inference_impl(
            init_date,
            outputs_repo_obj=local_outputs_repo,
            runner_factory=lambda: fake_runner,
            regridder_factory=lambda: identity_regridder,
            ic_dir=ic_dir,
            lead_time=12,
            chunk_layout=None,
        )
        assert ran is True
        sess = local_outputs_repo.readonly_session("main")
        ds = xr.open_dataset(
            sess.store,
            group=utils.datetime_to_str(init_date),
            engine="zarr",
            zarr_format=3,
            chunks=None,
        )
        assert ds.sizes["lead_time"] == 2
        assert ds.sizes["init_time"] == 1
        assert "10u" in ds.data_vars
        for d in [init_date - dt.timedelta(hours=6), init_date]:
            assert not os.path.exists(os.path.join(ic_dir, utils.datetime_to_str(d)))

    def test_keep_ics(
        self,
        ic_dir,
        local_outputs_repo,
        write_ic_dates,
        fake_runner,
        identity_regridder,
        init_date,
    ):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        _app._run_inference_impl(
            init_date,
            outputs_repo_obj=local_outputs_repo,
            runner_factory=lambda: fake_runner,
            regridder_factory=lambda: identity_regridder,
            ic_dir=ic_dir,
            lead_time=6,
            keep_ics=True,
            chunk_layout=None,
        )
        for d in [init_date - dt.timedelta(hours=6), init_date]:
            assert os.path.exists(os.path.join(ic_dir, utils.datetime_to_str(d)))

    def test_skip_when_already_exists(
        self,
        ic_dir,
        local_outputs_repo,
        fake_runner,
        identity_regridder,
        init_date,
        rng,
    ):
        sess = local_outputs_repo.writable_session("main")
        xr.Dataset({"x": (("a",), rng.random(3))}).to_zarr(
            sess.store,
            group=utils.datetime_to_str(init_date),
            zarr_format=3,
            consolidated=False,
            mode="w",
        )
        sess.commit("seed")

        runner_called = mock.Mock()

        def factory():
            runner_called()
            return fake_runner

        ran = _app._run_inference_impl(
            init_date,
            outputs_repo_obj=local_outputs_repo,
            runner_factory=factory,
            regridder_factory=lambda: identity_regridder,
            ic_dir=ic_dir,
        )
        assert ran is False
        runner_called.assert_not_called()

    def test_creates_missing_output_branch(
        self,
        ic_dir,
        local_outputs_repo,
        write_ic_dates,
        fake_runner,
        identity_regridder,
        init_date,
    ):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        _app._run_inference_impl(
            init_date,
            outputs_repo_obj=local_outputs_repo,
            runner_factory=lambda: fake_runner,
            regridder_factory=lambda: identity_regridder,
            ic_dir=ic_dir,
            outputs_branch="experiment-x",
            lead_time=6,
            chunk_layout=None,
        )
        assert "experiment-x" in local_outputs_repo.list_branches()

    def test_ensemble_concat(
        self,
        ic_dir,
        local_outputs_repo,
        write_ic_dates,
        fake_runner,
        identity_regridder,
        init_date,
    ):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        _app._run_inference_impl(
            init_date,
            outputs_repo_obj=local_outputs_repo,
            runner_factory=lambda: fake_runner,
            regridder_factory=lambda: identity_regridder,
            ic_dir=ic_dir,
            lead_time=6,
            n_members=3,
            chunk_layout=None,
        )
        sess = local_outputs_repo.readonly_session("main")
        ds = xr.open_dataset(
            sess.store,
            group=utils.datetime_to_str(init_date),
            engine="zarr",
            zarr_format=3,
            chunks=None,
        )
        assert ds.sizes["ensemble_member"] == 3

    def test_custom_chunk_layout_called(
        self,
        ic_dir,
        local_outputs_repo,
        write_ic_dates,
        fake_runner,
        identity_regridder,
        init_date,
    ):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        seen = []

        def layout(ds):
            seen.append(ds)
            return ds

        _app._run_inference_impl(
            init_date,
            outputs_repo_obj=local_outputs_repo,
            runner_factory=lambda: fake_runner,
            regridder_factory=lambda: identity_regridder,
            ic_dir=ic_dir,
            lead_time=6,
            chunk_layout=layout,
        )
        assert len(seen) == 1

    def test_on_ic_reload_invoked(
        self,
        ic_dir,
        local_outputs_repo,
        write_ic_dates,
        fake_runner,
        identity_regridder,
        init_date,
    ):
        write_ic_dates([init_date - dt.timedelta(hours=6), init_date])
        reload_mock = mock.Mock()
        _app._run_inference_impl(
            init_date,
            outputs_repo_obj=local_outputs_repo,
            runner_factory=lambda: fake_runner,
            regridder_factory=lambda: identity_regridder,
            ic_dir=ic_dir,
            on_ic_reload=reload_mock,
            lead_time=6,
            chunk_layout=None,
        )
        reload_mock.assert_called_once()


# ===========================================================================
# Modal-facing wrappers (modal mocked in conftest)
# ===========================================================================
class TestModalWrappers:
    def test_run_forecast_wires_modal_primitives(self, monkeypatch, init_date):
        captured = {}

        def fake_impl(date, bucket, **kwargs):
            captured["date"] = date
            captured.update(kwargs)

        monkeypatch.setattr(_app, "_run_forecast_impl", fake_impl)
        _app.run_forecast(init_date, "bucket", outputs_prefix="p")
        assert captured["date"] == init_date
        assert captured["ingest_era5_arco_fn"] is _app.ingest_era5_arco.remote
        assert captured["ingest_ifs_arraylake_fn"] is _app.ingest_ifs_arraylake.remote
        assert captured["run_inference_fn"] is _app.run_inference.remote
        assert captured["ic_dir"] == settings.IC_DIR
        assert captured["on_ic_reload"] is _app.ic_volume.reload

    def test_run_forecast_ekd_fn_ingests_and_commits(self, monkeypatch, init_date):
        captured = {}
        monkeypatch.setattr(
            _app, "_run_forecast_impl", lambda *a, **kw: captured.update(kw)
        )
        ekd_calls = []
        monkeypatch.setattr(_app.ingest_ekd, "ingest", lambda *a: ekd_calls.append(a))
        _app.run_forecast(init_date, "bucket", outputs_prefix="p")
        _app.ic_volume.commit.reset_mock()
        captured["ingest_ekd_fn"]("s", "e", "/ic", "ifs-ekd")
        assert ekd_calls == [("s", "e", "/ic", "ifs-ekd")]
        _app.ic_volume.commit.assert_called_once()

    def test_ingest_era5_arco_commits_volume(self, monkeypatch):
        calls = []
        monkeypatch.setattr(_ingest_era5_arco, "ingest", lambda *a: calls.append(a))
        _app.ic_volume.commit.reset_mock()
        _app.ingest_era5_arco("2024-01-01T00:00:00", "2024-01-01T06:00:00")
        assert calls == [
            ("2024-01-01T00:00:00", "2024-01-01T06:00:00", settings.IC_DIR)
        ]
        _app.ic_volume.commit.assert_called_once()

    @pytest.fixture
    def patched_inference(self, monkeypatch):
        repo = object()
        monkeypatch.setattr(_app, "_open_outputs_repo", lambda *a, **kw: repo)
        impl = mock.Mock(return_value=True)
        monkeypatch.setattr(_app, "_run_inference_impl", impl)
        _app.ic_volume.commit.reset_mock()
        return repo, impl

    def test_run_inference_commits_ics_after_write(self, patched_inference, init_date):
        repo, impl = patched_inference
        _app.run_inference(init_date, "bucket", {}, outputs_prefix="p")
        assert impl.call_args.kwargs["outputs_repo_obj"] is repo
        _app.ic_volume.commit.assert_called_once()

    def test_run_inference_no_commit_on_skip(self, patched_inference, init_date):
        _repo, impl = patched_inference
        impl.return_value = False
        _app.run_inference(init_date, "bucket", {}, outputs_prefix="p")
        _app.ic_volume.commit.assert_not_called()

    def test_run_inference_no_commit_with_keep_ics(self, patched_inference, init_date):
        _repo, impl = patched_inference
        _app.run_inference(init_date, "bucket", {}, outputs_prefix="p", keep_ics=True)
        assert impl.call_args.kwargs["keep_ics"] is True
        _app.ic_volume.commit.assert_not_called()


# ===========================================================================
# Public API surface
# ===========================================================================
class TestPublicAPI:
    def test_exports(self):
        import aifs_modal

        for name in (
            "app",
            "apply_trajectory_chunks",
            "forecast_exists",
            "ingest_era5_arco",
            "ingest_ifs_arraylake",
            "run_forecast",
            "settings",
        ):
            assert hasattr(aifs_modal, name)

    def test_settings_defaults(self):
        assert settings.DEFAULT_IC_SOURCE in settings.IC_SOURCES
        assert settings.DEFAULT_CHUNK_LAYOUT is utils.apply_trajectory_chunks
