from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from xregrid.utils import is_lazy, is_dask, is_cubed, create_grid_like
from xregrid import Regridder


def test_backend_utilities():
    """Verify backend-agnostic utilities."""
    # Eager
    ds_eager = xr.Dataset({"a": (["x"], np.arange(10))})
    assert not is_lazy(ds_eager)
    assert not is_dask(ds_eager)
    assert not is_cubed(ds_eager)

    # Dask
    ds_dask = ds_eager.chunk({"x": 5})
    assert is_lazy(ds_dask)
    assert is_dask(ds_dask)
    assert not is_cubed(ds_dask)

    # Individual arrays
    assert is_dask(ds_dask.a)
    assert not is_dask(ds_eager.a)


def test_create_grid_like_hardening():
    """Verify create_grid_like avoids computes when metadata is present."""
    ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-90, 91, 1.0)),
            "lon": (["lon"], np.arange(0, 361, 1.0)),
        }
    )
    ds.attrs["geospatial_lat_min"] = -90.0
    ds.attrs["geospatial_lat_max"] = 90.0
    ds.attrs["geospatial_lon_min"] = 0.0
    ds.attrs["geospatial_lon_max"] = 360.0

    # Chunk it to make it lazy
    ds_lazy = ds.chunk({"lat": 10, "lon": 10})

    # This should NOT trigger a compute if metadata discovery works
    # We can verify by checking if a warning is issued (we added warnings.warn)
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        grid = create_grid_like(ds_lazy, res=2.0)

        # Check that NO UserWarning from create_grid_like was issued
        # (The warning is issued only if it falls back to compute)
        for warning in record:
            if "Triggering hidden compute" in str(warning.message):
                pytest.fail(
                    "create_grid_like triggered a compute despite metadata presence"
                )

    assert grid.lat.size == 90
    assert grid.lon.size == 180


def test_lazy_diagnostics():
    """Verify diagnostics and quality_report preserve laziness."""
    src = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-10, 11, 2.0)),
            "lon": (["lon"], np.arange(0, 21, 2.0)),
        }
    )
    tgt = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-10, 11, 1.0)),
            "lon": (["lon"], np.arange(0, 21, 1.0)),
        }
    )

    # Use parallel=True to get distributed weights (simulated via LocalCluster if needed)
    # But for a simple test, we can just check the logic if we mock the matrix as having a 'key'
    regridder = Regridder(src, tgt, method="bilinear", parallel=True)

    # Ensure weights matrix is "remote" (mocking it if necessary, but parallel=True should do it)
    if not hasattr(regridder._weights_matrix, "key"):
        # If parallel=True didn't make it remote (e.g. no cluster), manually mock it for the test
        class MockRemote:
            def __init__(self, obj):
                self.obj = obj
                self.key = "mock_key"

            def sum(self, axis=None):
                return self.obj.sum(axis=axis)

            @property
            def shape(self):
                return self.obj.shape

            @property
            def nnz(self):
                return self.obj.nnz

        regridder._weights_matrix = MockRemote(regridder._weights_matrix)

    # 1. Diagnostics
    ds_diag = regridder.diagnostics()
    assert is_dask(ds_diag.weight_sum)
    assert is_dask(ds_diag.unmapped_mask)

    # 2. Quality Report
    ds_report = regridder.quality_report(format="dataset")
    assert isinstance(ds_report, xr.Dataset)
    # n_weights should be lazy if format='dataset' and remote
    assert is_dask(ds_report.n_weights)
    assert is_dask(ds_report.weight_sum_max)


def test_double_check_logic():
    """Verify consistency between Eager and Lazy applications (Aero Protocol)."""
    src = xr.Dataset(
        coords={
            "lat": (["lat"], np.linspace(-10, 10, 5)),
            "lon": (["lon"], np.linspace(0, 20, 5)),
        }
    )
    tgt = xr.Dataset(
        coords={
            "lat": (["lat"], np.linspace(-10, 10, 10)),
            "lon": (["lon"], np.linspace(0, 20, 10)),
        }
    )

    data = np.random.rand(5, 5)
    da_eager = xr.DataArray(data, dims=["lat", "lon"], coords=src.coords, name="test")
    da_lazy = da_eager.chunk({"lat": 2, "lon": 2})

    regridder = Regridder(src, tgt, method="bilinear")

    res_eager = regridder(da_eager)
    res_lazy = regridder(da_lazy)

    assert is_dask(res_lazy)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Verify provenance
    assert "backend=Eager" in res_eager.attrs["history"]
    assert "backend=Distributed (Dask)" in res_lazy.attrs["history"]
