from __future__ import annotations

import xarray as xr

from xregrid.utils import create_global_grid, create_grid_from_crs, is_dask


def test_create_global_grid_parity():
    """Verify parity between eager and lazy global grid creation."""
    # 1. Eager (NumPy)
    ds_eager = create_global_grid(res_lat=10, res_lon=10, add_bounds=True)

    # 2. Lazy (Dask)
    ds_lazy = create_global_grid(res_lat=10, res_lon=10, add_bounds=True, chunks=10)

    assert is_dask(ds_lazy)
    assert not is_dask(ds_eager)

    # Check parity of coordinates
    xr.testing.assert_allclose(ds_eager.lat, ds_lazy.lat.compute())
    xr.testing.assert_allclose(ds_eager.lon, ds_lazy.lon.compute())
    xr.testing.assert_allclose(ds_eager.lat_b, ds_lazy.lat_b.compute())
    xr.testing.assert_allclose(ds_eager.lon_b, ds_lazy.lon_b.compute())


def test_create_grid_from_crs_parity():
    """Verify parity between eager and lazy projected grid creation."""
    crs = "EPSG:3857"
    extent = (-1000000, 1000000, -1000000, 1000000)
    res = 500000

    # 1. Eager
    ds_eager = create_grid_from_crs(crs, extent, res, add_bounds=True)

    # 2. Lazy
    ds_lazy = create_grid_from_crs(crs, extent, res, add_bounds=True, chunks=2)

    assert is_dask(ds_lazy)

    # Check parity
    xr.testing.assert_allclose(ds_eager.x, ds_lazy.x.compute())
    xr.testing.assert_allclose(ds_eager.y, ds_lazy.y.compute())
    xr.testing.assert_allclose(ds_eager.lat, ds_lazy.lat.compute())
    xr.testing.assert_allclose(ds_eager.lon, ds_lazy.lon.compute())

    if "lat_b" in ds_eager:
        xr.testing.assert_allclose(ds_eager.lat_b, ds_lazy.lat_b.compute())
        xr.testing.assert_allclose(ds_eager.lon_b, ds_lazy.lon_b.compute())

    if "x_b" in ds_eager:
        xr.testing.assert_allclose(ds_eager.x_b, ds_lazy.x_b.compute())
        xr.testing.assert_allclose(ds_eager.y_b, ds_lazy.y_b.compute())


def test_provenance_tracking():
    """Verify that history attribute is updated."""
    ds = create_global_grid(res_lat=10, res_lon=10)
    assert "history" in ds.attrs
    assert "xregrid" in ds.attrs["history"]
