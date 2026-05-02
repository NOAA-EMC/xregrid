from __future__ import annotations
import numpy as np
import xarray as xr
from xregrid.utils import create_global_grid, create_mesh_from_coords, spatial_slice


def test_spatial_slice_rectilinear() -> None:
    """
    Test spatial_slice with a rectilinear grid (Eager and Lazy).

    Verifies that spatial_slice correctly subsets a rectilinear grid
    using coordinate indexes and handles both NumPy and Dask backends.
    """
    # 1. Eager
    ds = create_global_grid(res_lat=1.0, res_lon=1.0)
    # Extent: (min_x, max_x, min_y, max_y)
    extent = (10.5, 20.5, 30.5, 40.5)
    ds_sliced = spatial_slice(ds, extent)

    assert ds_sliced.lat.min() >= 30.5
    assert ds_sliced.lat.max() <= 40.5
    assert ds_sliced.lon.min() >= 10.5
    assert ds_sliced.lon.max() <= 20.5
    assert "history" in ds_sliced.attrs
    assert "Spatially sliced" in ds_sliced.attrs["history"]

    # 2. Lazy (Dask)
    ds_lazy = create_global_grid(
        res_lat=1.0, res_lon=1.0, chunks={"lat": 10, "lon": 10}
    )
    ds_sliced_lazy = spatial_slice(ds_lazy, extent)

    # In xarray, dimension coordinates are often eager (NumPy) due to indexing.
    # Check lat_b instead, which should remain lazy.
    assert hasattr(ds_sliced_lazy.lat_b.data, "dask")
    xr.testing.assert_allclose(ds_sliced, ds_sliced_lazy.compute())


def test_spatial_slice_unstructured() -> None:
    """
    Test spatial_slice with an unstructured grid (Eager and Lazy).

    Verifies that spatial_slice correctly subsets an unstructured grid
    using boolean masking and maintains laziness for Dask-backed data.
    """
    # Create points: one in the box, one outside
    lons = np.array([15.0, 50.0])
    lats = np.array([35.0, 60.0])
    ds = create_mesh_from_coords(lons, lats, crs="EPSG:4326")

    extent = (10.0, 20.0, 30.0, 40.0)

    # Eager path: drop=True is supported
    ds_sliced = spatial_slice(ds, extent)

    assert ds_sliced.sizes["n_pts"] == 1
    assert ds_sliced.lon.values[0] == 15.0
    assert ds_sliced.lat.values[0] == 35.0

    # Lazy path: drop=False is used to preserve laziness
    ds_lazy = create_mesh_from_coords(lons, lats, crs="EPSG:4326", chunks=1)
    ds_lazy["data"] = (["n_pts"], ds_lazy.lat.data, {"units": "K"})
    ds_sliced_lazy = spatial_slice(ds_lazy, extent)

    # Verify laziness
    assert hasattr(ds_sliced_lazy.data.data, "dask")

    # Verify results (after compute and dropna since drop=False was used)
    ds_res = ds_sliced_lazy.compute().dropna("n_pts")
    assert ds_res.sizes["n_pts"] == 1
    assert ds_res.lon.values[0] == 15.0


def test_spatial_slice_wrapping() -> None:
    """
    Test spatial_slice with longitude wrapping across different grid types.

    Verifies that spatial_slice correctly handles regions crossing the
    meridian/dateline for both rectilinear and unstructured grids.
    """
    ds = create_global_grid(res_lat=1.0, res_lon=1.0)

    # Slice crossing the 0/360 boundary
    # Extent: (min_x, max_x, min_y, max_y)
    extent = (-10.5, 10.5, -10.5, 10.5)
    ds_sliced = spatial_slice(ds, extent)

    # Lon in ds is 0-360. -10.5 should map to 349.5
    assert ds_sliced.lon.min() >= 0
    assert ds_sliced.lon.max() <= 360

    # Check that we have both the 0-10.5 and 349.5-360 parts
    assert (ds_sliced.lon <= 10.5).any()
    assert (ds_sliced.lon >= 349.5).any()

    # Verify unstructured wrapping
    lons = np.array([5.0, 355.0, 180.0])
    lats = np.array([0.0, 0.0, 0.0])
    ds_unstructured = create_mesh_from_coords(lons, lats, crs="EPSG:4326")
    ds_un_sliced = spatial_slice(ds_unstructured, extent)

    # Since it's eager, drop=True was used
    assert ds_un_sliced.sizes["n_pts"] == 2
    assert set(ds_un_sliced.lon.values) == {5.0, 355.0}
