"""Shared fixtures and module-level mocks for the aifs_modal test suite.

Mocks `modal` at import time so `aifs_modal._app` (which calls
`modal.Volume.from_name`, `modal.App(...)`, etc. at top level) imports
cleanly on CI without Modal credentials.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Modal mock — installed before aifs_modal imports anywhere
# ---------------------------------------------------------------------------
def _install_modal_mock() -> None:
    if "modal" in sys.modules:
        return

    m = types.ModuleType("modal")

    class _Volume:
        @staticmethod
        def from_name(name, create_if_missing=False):
            v = mock.Mock()
            v.reload = mock.Mock()
            v.commit = mock.Mock()
            v.hydrate = mock.Mock()
            return v

    class _Secret:
        @staticmethod
        def from_name(name):
            return mock.Mock()

    def _fake_image():
        img = mock.Mock()
        for name in ("uv_pip_install", "apt_install", "env", "run_commands"):
            setattr(img, name, mock.Mock(return_value=img))
        return img

    class _Image:
        @staticmethod
        def debian_slim(**_):
            return _fake_image()

        @staticmethod
        def from_registry(_n, **_kw):
            return _fake_image()

    class _App:
        def __init__(self, name):
            self.name = name

        def function(self, **_kw):
            def deco(fn):
                # Mirror modal's `.remote` attribute used in the orchestrator.
                fn.remote = mock.Mock(name=f"{fn.__name__}.remote")
                fn._modal_stub = True
                return fn

            return deco

        def local_entrypoint(self, **_kw):
            def deco(fn):
                return fn

            return deco

    m.Volume, m.Secret, m.Image, m.App = _Volume, _Secret, _Image, _App
    sys.modules["modal"] = m


_install_modal_mock()


# ---------------------------------------------------------------------------
# torch + anemoi stubs so _run_member / _run_inference_impl can execute
# without the real GPU dependencies installed.
# ---------------------------------------------------------------------------
def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    t = types.ModuleType("torch")
    t.manual_seed = lambda _s: None
    t.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    sys.modules["torch"] = t


def _install_anemoi_stub() -> None:
    for name in (
        "anemoi",
        "anemoi.inference",
        "anemoi.inference.outputs",
        "anemoi.inference.outputs.printer",
        "anemoi.inference.runners",
        "anemoi.inference.runners.simple",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["anemoi.inference.outputs.printer"].print_state = lambda _s: None
    sys.modules["anemoi.inference.runners.simple"].SimpleRunner = mock.Mock(
        name="SimpleRunner"
    )


_install_torch_stub()
_install_anemoi_stub()


# ---------------------------------------------------------------------------
# Shrink the output grid so synthetic fields can be tiny.
# state_to_xarray references _app.LAT / _app.LON at call time, so patching
# the module globals is sufficient.
# ---------------------------------------------------------------------------
N_LAT = 4
N_LON = 8
N_FLAT = N_LAT * N_LON


@pytest.fixture(autouse=True, scope="session")
def _shrink_output_grid():
    from aifs_modal import _app

    saved = (_app.LAT, _app.LON)
    _app.LAT = 90 - (180 / N_LAT) * np.arange(N_LAT)
    _app.LON = (360 / N_LON) * np.arange(N_LON)
    yield
    _app.LAT, _app.LON = saved


# ---------------------------------------------------------------------------
# Time + random
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    """Return a seeded numpy random generator for reproducible test data."""
    return np.random.default_rng(0)


@pytest.fixture
def init_date():
    """Return a valid 6h-aligned UTC datetime used by many tests."""
    return dt.datetime(2024, 1, 1, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# IC store on tmp_path
# ---------------------------------------------------------------------------
@pytest.fixture
def ic_dir(tmp_path):
    """Return an empty initial-conditions directory under tmp_path."""
    d = tmp_path / "ic"
    d.mkdir()
    return str(d)


@pytest.fixture
def write_ic_dates(ic_dir, rng):
    """Write minimal synthetic IC zarr groups for the given dates.

    Variable names are chosen so :func:`_ic.fetch_initial_conditions` returns
    everything the fake AIFS runner expects after the gh→z transform and the
    name re-mapping.
    """
    from aifs_modal import _ic

    def _write(dates):
        for d in dates:
            data = {
                # surface, no rename
                "10u": rng.random(8, dtype="f4"),
                "10v": rng.random(8, dtype="f4"),
                "2t": rng.random(8, dtype="f4"),
                # gets renamed by fetch_initial_conditions
                "tcwv": rng.random(8, dtype="f4"),
                "sot_1": rng.random(8, dtype="f4"),
                "vsw_1": rng.random(8, dtype="f4"),
            }
            # gh_<level> -> z_<level> conversion needs all levels present
            for lvl in _ic.LEVELS:
                data[f"gh_{lvl}"] = rng.random(8, dtype="f4")
            _ic.get_and_store_date(d, ic_dir, lambda _date, _d=data: _d)

    return _write


# ---------------------------------------------------------------------------
# State / runner / regridder fixtures for inference paths
# ---------------------------------------------------------------------------
@pytest.fixture
def identity_regridder():
    """Reshape (N_FLAT,) -> (N_LAT, N_LON). Stands in for the GPU regridder."""

    class _R:
        target_shape = (N_LAT, N_LON)

        def regrid(self, arr):
            return np.asarray(arr, dtype="f4").reshape(N_LAT, N_LON)

    return _R()


def _full_aifs_field_set(rng):
    """Build a dict with all surface + all pressure-level vars used by AIFS, flat."""
    from aifs_modal._app import PRESSURE_LEVELS, PRESSURE_VAR_PREFIXES

    surface = [
        "10u",
        "10v",
        "2d",
        "2t",
        "msl",
        "skt",
        "sp",
        "tcw",
        "lsm",
        "z",
        "slor",
        "sdor",
        "swvl1",
        "swvl2",
        "stl1",
        "stl2",
    ]
    fields = {n: rng.random(N_FLAT, dtype="f4") for n in surface}
    for p in PRESSURE_VAR_PREFIXES:
        for lvl in PRESSURE_LEVELS:
            fields[f"{p}_{lvl}"] = rng.random(N_FLAT, dtype="f4")
    return fields


@pytest.fixture
def aifs_fields(rng):
    """Return the full AIFS field dict as flat synthetic arrays."""
    return _full_aifs_field_set(rng)


@pytest.fixture
def fake_runner(rng):
    """Stand-in for ``anemoi.inference.runners.simple.SimpleRunner``.

    Yields states every 6h up to ``lead_time``, with the full AIFS field
    schema so :func:`state_to_xarray` can drop the pressure-level vars.
    """
    expected = list(_full_aifs_field_set(rng).keys())

    class _Runner:
        checkpoint = SimpleNamespace(
            variable_to_input_tensor_index={v: i for i, v in enumerate(expected)}
        )

        def run(self, input_state, lead_time):
            start = input_state["date"]
            local_rng = np.random.default_rng(123)
            for h in range(6, (lead_time or 6) + 1, 6):
                yield {
                    "date": start + dt.timedelta(hours=h),
                    "fields": {
                        n: local_rng.random(N_FLAT, dtype="f4") for n in expected
                    },
                }

    return _Runner()


# ---------------------------------------------------------------------------
# Local-filesystem icechunk repo (no S3, no moto)
# ---------------------------------------------------------------------------
@pytest.fixture
def local_outputs_repo(tmp_path):
    """Return a local-filesystem icechunk repo standing in for the S3 outputs repo."""
    import icechunk

    storage = icechunk.local_filesystem_storage(str(tmp_path / "outputs"))
    return icechunk.Repository.open_or_create(storage)
