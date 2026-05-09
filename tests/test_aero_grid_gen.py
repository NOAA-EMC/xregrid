import numpy as np
import xarray as xr
import pytest
from xregrid.utils import (
    create_global_grid,
    create_regional_grid,
    create_grid_from_crs,
    create_mesh_from_coords,
)


def test_global_grid_backend_consistency():
    """Verify that global grid generation yields identical results for NumPy and Dask."""
    res = 1.0
    ds_eager = create_global_grid(res, res, chunks=None)
    ds_lazy = create_global_grid(res, res, chunks={"lat": 10, "lon": 10})

    # Verify values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify backend
    assert not hasattr(ds_eager.lat.data, "dask")
    # In recent xarray, coordinate indexes are eager-loaded into NumPy.
    # We check non-index coords for laziness.
    assert hasattr(ds_lazy.lat_b.data, "dask")
    assert hasattr(ds_lazy.lon_b.data, "dask")


def test_regional_grid_backend_consistency():
    """Verify that regional grid generation yields identical results for NumPy and Dask."""
    lat_range = (10, 20)
    lon_range = (30, 40)
    res = 0.5
    ds_eager = create_regional_grid(lat_range, lon_range, res, res, chunks=None)
    ds_lazy = create_regional_grid(lat_range, lon_range, res, res, chunks=10)

    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_grid_from_crs_backend_consistency():
    """Verify that CRS-based grid generation yields identical results for NumPy and Dask."""
    pytest.importorskip("pyproj")
    crs = "EPSG:32633"  # UTM zone 33N
    extent = (400000, 500000, 5000000, 5100000)
    res = 10000

    ds_eager = create_grid_from_crs(crs, extent, res, chunks=None)
    ds_lazy = create_grid_from_crs(crs, extent, res, chunks={"x": 5, "y": 5})

    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Check that heavy 2D coordinates are lazy
    assert hasattr(ds_lazy.lat.data, "dask")
    assert hasattr(ds_lazy.lon.data, "dask")
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_laziness_large_grid():
    """Verify that creating a massive grid doesn't crash the driver (truly lazy)."""
    # 0.001 degree global grid would be 180,000 x 360,000 points.
    # In NumPy, this would be ~500 GB for just one coordinate array.
    # If this crash/hangs, the logic is not lazy.
    res = 0.001
    ds_lazy = create_global_grid(res, res, chunks={"lat": 1000, "lon": 1000})

    # Verify it is lazy and hasn't allocated the full array
    assert hasattr(ds_lazy.lat_b.data, "dask")
    # Check shape to ensure it's correct
    assert ds_lazy.lat.size == 180000
    assert ds_lazy.lon.size == 360000


def test_mesh_laziness_backend_consistency():
    """
    Aero Protocol: Double-Check test for mesh laziness.
    Verifies that coordinates are lazy when chunks are provided and
    results are identical between Eager and Lazy backends.
    """
    pytest.importorskip("pyproj")
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


def test_mesh_laziness_dict_chunks():
    """Verify that dict-based chunks are also handled correctly for n_pts."""
    pytest.importorskip("pyproj")
    n_pts = 500
    x = np.linspace(-10, 10, n_pts)
    y = np.linspace(-10, 10, n_pts)
    crs = "EPSG:4326"

    chunks = {"n_pts": 250}
    ds_lazy = create_mesh_from_coords(x, y, crs=crs, chunks=chunks)

    assert hasattr(ds_lazy.x.data, "dask")
    assert ds_lazy.x.chunks[0][0] == 250
