"""Tests for AIFS on Modal."""

import datetime
import unittest
from unittest import mock

import icechunk
import numpy as np
import zarr

from aifs_modal import utils
from aifs_modal.ic import (
    LEVELS,
    _iter_dates_6h,
    _parse_utc_date,
    fetch_initial_conditions,
    ingest_range,
)
from aifs_modal.ic import (
    _stack_fields as stack_fields,
)
from aifs_modal.ic import (
    _store_data as store_data,
)
from aifs_modal.ingest_ekd import (
    PARAM_PL,
    PARAM_SFC,
    PARAM_SOIL,
    SOIL_LEVELS,
)


class TestImports(unittest.TestCase):
    def test_imports(self):
        import aifs_modal  # noqa: F401


class TestParseUtcDate(unittest.TestCase):
    def test_naive_gets_utc(self):
        dt = _parse_utc_date("2025-01-01T00:00:00")
        self.assertEqual(dt.tzinfo, datetime.UTC)

    def test_aware_converted_to_utc(self):
        dt = _parse_utc_date("2025-01-01T00:00:00+00:00")
        self.assertEqual(dt.tzinfo, datetime.UTC)

    def test_valid_hours(self):
        for hour in [0, 6, 12, 18]:
            dt = _parse_utc_date(f"2025-06-20T{hour:02d}:00:00")
            self.assertEqual(dt.hour, hour)

    def test_invalid_hour_raises(self):
        with self.assertRaises(ValueError):
            _parse_utc_date("2025-01-01T03:00:00")

    def test_minutes_raises(self):
        with self.assertRaises(ValueError):
            _parse_utc_date("2025-01-01T00:30:00")


class TestIterDates6h(unittest.TestCase):
    def test_single_date(self):
        dt = datetime.datetime(2025, 1, 1, 0, tzinfo=datetime.UTC)
        self.assertEqual(list(_iter_dates_6h(dt, dt)), [dt])

    def test_range(self):
        start = datetime.datetime(2025, 1, 1, 0, tzinfo=datetime.UTC)
        end = datetime.datetime(2025, 1, 1, 18, tzinfo=datetime.UTC)
        expected = [
            datetime.datetime(2025, 1, 1, h, tzinfo=datetime.UTC)
            for h in [0, 6, 12, 18]
        ]
        self.assertEqual(list(_iter_dates_6h(start, end)), expected)

    def test_end_before_start_is_empty(self):
        start = datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC)
        end = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
        self.assertEqual(list(_iter_dates_6h(start, end)), [])


class TestStackFields(unittest.TestCase):
    def test_names_and_shape(self):
        npoints = 100
        rng = np.random.default_rng(0)
        data = {k: rng.random(npoints).astype("f4") for k in ["a", "b", "c"]}
        names, stacked = stack_fields(data)
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(stacked.shape, (3, npoints))

    def test_values_preserved(self):
        npoints = 50
        rng = np.random.default_rng(1)
        data = {
            "x": rng.random(npoints).astype("f4"),
            "y": rng.random(npoints).astype("f4"),
        }
        names, stacked = stack_fields(data)
        np.testing.assert_array_equal(stacked[0], data["x"])
        np.testing.assert_array_equal(stacked[1], data["y"])


class TestStoreData(unittest.TestCase):
    def _make_group(self):
        store = zarr.storage.MemoryStore()
        return zarr.group(store=store, zarr_format=3)

    def test_roundtrip(self):
        group = self._make_group()
        nvars, npoints = 4, 100
        names = ["a", "b", "c", "d"]
        rng = np.random.default_rng(0)
        data = rng.random((nvars, npoints)).astype("f4")
        store_data(group, names, data)
        np.testing.assert_array_equal(group["variable"][:], names)
        np.testing.assert_array_equal(group["fields"][:], data)


class TestFetchInitialConditions(unittest.TestCase):
    NPOINTS = 10

    def _make_synthetic_fields(self, seed):
        rng = np.random.default_rng(seed)
        fields = {}
        for param in PARAM_SFC:
            fields[param] = rng.random(self.NPOINTS).astype("f4")
        for param in PARAM_SOIL:
            for level in SOIL_LEVELS:
                fields[f"{param}_{level}"] = rng.random(self.NPOINTS).astype("f4")
        for param in PARAM_PL:
            for level in LEVELS:
                fields[f"{param}_{level}"] = rng.random(self.NPOINTS).astype("f4")
        return fields

    def _populate_group(self, session, date, fields):
        group_name = utils.datetime_to_str(date)
        group = zarr.group(store=session.store, path=group_name, overwrite=True)
        names, stacked = stack_fields(fields)
        store_data(group, names, stacked)

    def setUp(self):
        storage = icechunk.in_memory_storage()
        repo = icechunk.Repository.create(storage)
        self._session = repo.writable_session("main")

        self._date = datetime.datetime(2025, 6, 20, 0, tzinfo=datetime.UTC)
        self._date_prev = self._date - datetime.timedelta(hours=6)
        self._fields_prev = self._make_synthetic_fields(seed=42)
        self._fields_curr = self._make_synthetic_fields(seed=43)

        self._populate_group(self._session, self._date_prev, self._fields_prev)
        self._populate_group(self._session, self._date, self._fields_curr)

    def test_variable_renaming(self):
        result = fetch_initial_conditions(self._date, self._session)
        self.assertIn("tcw", result)
        self.assertNotIn("tcwv", result)
        self.assertIn("stl1", result)
        self.assertNotIn("sot_1", result)
        self.assertIn("stl2", result)
        self.assertNotIn("sot_2", result)
        self.assertIn("swvl1", result)
        self.assertNotIn("vsw_1", result)
        self.assertIn("swvl2", result)
        self.assertNotIn("vsw_2", result)

    def test_gh_to_z_conversion(self):
        result = fetch_initial_conditions(self._date, self._session)
        for level in LEVELS:
            self.assertIn(f"z_{level}", result)
            self.assertNotIn(f"gh_{level}", result)
            # index 0 = t-6h, index 1 = t
            np.testing.assert_allclose(
                result[f"z_{level}"][0],
                self._fields_prev[f"gh_{level}"] * 9.80665,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                result[f"z_{level}"][1],
                self._fields_curr[f"gh_{level}"] * 9.80665,
                rtol=1e-5,
            )

    def test_data_shape(self):
        result = fetch_initial_conditions(self._date, self._session)
        for v in result.values():
            # each field has shape (2, npoints) — stacked [t-6h, t]
            self.assertEqual(v.shape, (2, self.NPOINTS))


class TestIngestRange(unittest.TestCase):
    def test_skips_existing_dates(self):
        storage = icechunk.in_memory_storage()
        calls = []

        def fetch_fn(date):
            calls.append(date)
            return {"a": np.arange(4, dtype="f4")}

        kwargs = dict(
            start_date="2025-01-01T00:00:00+00:00",
            end_date="2025-01-01T00:00:00+00:00",
            storage_bucket="unused",
            fetch_fn=fetch_fn,
            source="ifs-ekd",
            initial_conditions_prefix="unused",
        )

        with mock.patch("aifs_modal.ic.utils.get_storage", return_value=storage):
            ingest_range(**kwargs)
            ingest_range(**kwargs)

        self.assertEqual(len(calls), 1)
