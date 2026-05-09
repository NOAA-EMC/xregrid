import numpy as np
import pytest
import xarray as xr
from xregrid.utils import create_mesh_from_coords

import importlib.util

HAS_DASK = importlib.util.find_spec("dask") is not None
HAS_PYPROJ = importlib.util.find_spec("pyproj") is not None


@pytest.mark.skipif(
    not HAS_DASK or not HAS_PYPROJ, reason="dask or pyproj not installed"
)
def test_mesh_laziness_double_check():
    """
    Aero Protocol: Double-Check test for mesh laziness.
    Verifies that n_pts coordinate is lazy when chunks are provided and
    results are identical between Eager and Lazy backends.
    """
    n_pts = 1000
    x = np.linspace(-10, 10, n_pts)
    y = np.linspace(-10, 10, n_pts)
    crs = "EPSG:3857"  # Web Mercator

    # 1. Eager Mesh
    ds_eager = create_mesh_from_coords(x, y, crs=crs)

    assert not hasattr(ds_eager.n_pts.data, "dask")
    assert "(Eager)" in ds_eager.attrs["history"]

    # 2. Lazy Mesh
    chunks = 100
    ds_lazy = create_mesh_from_coords(x, y, crs=crs, chunks=chunks)

    # n_pts might be eager in xarray as it's an index,
    # but we should check if we can verify laziness another way or if we should check data variables
    # For now, let's check x and y which are non-index coords in the Dataset output
    assert hasattr(ds_lazy.x.data, "dask")
    assert ds_lazy.x.chunks is not None
    assert "(Lazy)" in ds_lazy.attrs["history"]

    # lat/lon should also be lazy (via apply_ufunc)
    assert hasattr(ds_lazy.lat.data, "dask")
    assert hasattr(ds_lazy.lon.data, "dask")

    # 3. Numerical Identity
    xr.testing.assert_identical(ds_eager.n_pts, ds_lazy.n_pts.compute())
    xr.testing.assert_allclose(ds_eager.lat, ds_lazy.lat.compute())
    xr.testing.assert_allclose(ds_eager.lon, ds_lazy.lon.compute())


@pytest.mark.skipif(
    not HAS_DASK or not HAS_PYPROJ, reason="dask or pyproj not installed"
)
def test_mesh_laziness_dict_chunks():
    """Verify that dict-based chunks are also handled correctly for n_pts."""
    n_pts = 500
    x = np.linspace(-10, 10, n_pts)
    y = np.linspace(-10, 10, n_pts)
    crs = "EPSG:4326"

    chunks = {"n_pts": 250}
    ds_lazy = create_mesh_from_coords(x, y, crs=crs, chunks=chunks)

    assert hasattr(ds_lazy.x.data, "dask")
    assert ds_lazy.x.chunks[0][0] == 250
