import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_eager_lazy_identity_dim_orders():
    """Verify Eager and Lazy results are identical even with different dimension orders."""
    # Source grid: 10x20
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20)

    # Target grid: 15x25
    lat_out = np.linspace(-90, 90, 15)
    lon_out = np.linspace(0, 360, 25)

    src_grid = xr.Dataset(coords={"lat": lat, "lon": lon})
    tgt_grid = xr.Dataset(coords={"lat": lat_out, "lon": lon_out})

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # 1. Eager (lat, lon)
    data = np.random.rand(10, 20)
    da_eager = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    res_eager = regridder(da_eager)

    # 2. Lazy (lon, lat) - different order!
    da_lazy = xr.DataArray(
        data.T, dims=("lon", "lat"), coords={"lat": lat, "lon": lon}
    ).chunk({"lon": 10, "lat": 5})
    res_lazy = regridder(da_lazy).compute()

    # Transpose back for comparison if needed, or check if xregrid handles it
    # xregrid's _regrid_dataarray uses input_core_dims=self._dims_source
    # which will handle the transposition automatically via apply_ufunc.

    xr.testing.assert_allclose(res_eager, res_lazy)
    assert isinstance(regridder(da_lazy).data, da.Array)


def test_skipna_robustness():
    """Verify skipna=True handles NaNs correctly in both Eager and Lazy paths."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)

    regridder = Regridder(src_grid, tgt_grid, method="bilinear", skipna=True)

    data = np.ones((18, 36))
    data[0, 0] = np.nan  # Put a NaN

    # Only use 'lat' and 'lon' coords, skip 'lat_b' and 'lon_b'
    da_coords = {c: src_grid.coords[c] for c in ["lat", "lon"]}
    da_eager = xr.DataArray(data, dims=("lat", "lon"), coords=da_coords)
    res_eager = regridder(da_eager)

    # The result at (0,0) should not be NaN if there are other valid points in the stencil
    # (depending on the method and stencil size)
    # But more importantly, eager and lazy should match.

    da_lazy = da_eager.chunk({"lat": 9, "lon": 18})
    res_lazy = regridder(da_lazy).compute()

    xr.testing.assert_allclose(res_eager, res_lazy)


def test_provenance_tracking():
    """Verify that history is correctly updated and preserved."""
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(20, 20)

    regridder = Regridder(src_grid, tgt_grid)

    da_coords = {c: src_grid.coords[c] for c in ["lat", "lon"]}
    da = xr.DataArray(np.random.rand(6, 12), dims=("lat", "lon"), coords=da_coords)
    da.attrs["history"] = "Original data"

    res = regridder(da)

    assert "history" in res.attrs
    assert "Original data" in res.attrs["history"]
    assert "Regridded" in res.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
