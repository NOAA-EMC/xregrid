from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder
from xregrid.utils import create_global_grid


def test_aero_protocol_equivalence():
    """
    Verify that regridding produces identical results for NumPy and Dask backends.
    Following the Aero Protocol: Flexibility rule.
    """
    # 1. Setup grids
    ds_src = create_global_grid(1.0, 1.0)
    ds_tgt = create_global_grid(2.0, 2.0)

    # Create source data with some pattern and NaNs
    data = np.sin(np.deg2rad(ds_src.lat)) * np.cos(np.deg2rad(ds_src.lon))
    # Coordinates for the DataArray should only include relevant dimensions
    coords = {
        k: v for k, v in ds_src.coords.items() if set(v.dims).issubset({"lat", "lon"})
    }
    da_numpy = xr.DataArray(data, coords=coords, dims=("lat", "lon"), name="test_data")

    # Add some NaNs to test skipna
    da_numpy.values[10:20, 10:20] = np.nan

    # 2. Setup Regridder
    regridder = Regridder(ds_src, ds_tgt, method="bilinear", skipna=True)

    # 3. Eager (NumPy) regridding
    res_numpy = regridder(da_numpy)

    # 4. Lazy (Dask) regridding
    da_dask = da_numpy.chunk({"lat": 45, "lon": 90})
    res_dask = regridder(da_dask)

    # Verify result is still lazy
    assert hasattr(res_dask.data, "dask")

    # Compute and compare
    res_dask_computed = res_dask.compute()

    xr.testing.assert_allclose(res_numpy, res_dask_computed)

    # 5. Scientific Hygiene: Check history attribute
    assert "history" in res_numpy.attrs
    assert "Regridded using xregrid.Regridder" in res_numpy.attrs["history"]
    assert "backend=Eager" in res_numpy.attrs["history"]

    assert "history" in res_dask_computed.attrs
    assert "backend=Distributed (Dask)" in res_dask_computed.attrs["history"]


def test_non_spatial_preservation():
    """
    Verify that non-spatial dimensions are correctly identified and preserved.
    """
    from xregrid.grid import _get_non_spatial_dims

    # Create dataset with various dimension names
    ds = xr.Dataset(
        data_vars={
            "temp": (("time", "lev", "lat", "lon"), np.random.rand(2, 5, 10, 20))
        },
        coords={
            "time": np.arange(2),
            "lev": np.arange(5),
            "lat": np.arange(10),
            "lon": np.arange(20),
        },
    )

    non_spatial = _get_non_spatial_dims(ds)
    assert "time" in non_spatial
    assert "lev" in non_spatial
    assert "lat" not in non_spatial
    assert "lon" not in non_spatial

    # Test with standard names
    ds_std = xr.Dataset(
        data_vars={
            "temp": (
                ("custom_time", "custom_p", "lat", "lon"),
                np.random.rand(2, 5, 10, 20),
            )
        }
    )
    ds_std["custom_time"] = (("custom_time"), np.arange(2), {"standard_name": "time"})
    ds_std["custom_p"] = (("custom_p"), np.arange(5), {"standard_name": "air_pressure"})

    non_spatial_std = _get_non_spatial_dims(ds_std)
    assert "custom_time" in non_spatial_std
    assert "custom_p" in non_spatial_std


if __name__ == "__main__":
    pytest.main([__file__])
