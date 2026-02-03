import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_cf_aware_dimension_mapping():
    """Verify that Regridder handles non-standard dimension names via CF-awareness."""
    # 1. Source grid with standard 'lat'/'lon'
    src_res = 10.0
    src_grid = create_global_grid(res_lat=src_res, res_lon=src_res)

    # 2. Target grid
    tgt_res = 5.0
    tgt_grid = create_global_grid(res_lat=tgt_res, res_lon=tgt_res)

    # 3. Initialize Regridder
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # 4. Input DataArray with different names: 'latitude' and 'longitude'
    # but marked with proper CF attributes
    data = np.random.rand(18, 36)
    da = xr.DataArray(
        data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": (
                ["latitude"],
                src_grid.lat.values,
                {"standard_name": "latitude"},
            ),
            "longitude": (
                ["longitude"],
                src_grid.lon.values,
                {"standard_name": "longitude"},
            ),
        },
        name="test_data",
    )

    # 5. Eager Regridding
    res_eager = regridder(da)

    assert res_eager.shape == (36, 72)
    assert res_eager.name == "test_data"

    # 6. Lazy Regridding (Double-Check Rule)
    da_lazy = da.chunk({"latitude": 9, "longitude": 18})
    res_lazy = regridder(da_lazy).compute()

    # 7. Verification
    xr.testing.assert_allclose(res_eager, res_lazy)

    # Verify coordinates match target grid
    np.testing.assert_allclose(res_eager.lat, tgt_grid.lat)
    np.testing.assert_allclose(res_eager.lon, tgt_grid.lon)


def test_dataset_cf_awareness():
    """Verify CF-aware regridding for multiple variables in a Dataset."""
    src_grid = create_global_grid(20, 20)
    tgt_grid = create_global_grid(10, 10)

    regridder = Regridder(src_grid, tgt_grid)

    # Dataset with mixed naming
    ds = xr.Dataset(
        data_vars={
            "temp": (("latitude", "longitude"), np.random.rand(9, 18)),
            "scalar": 42.0,
        },
        coords={
            "latitude": (
                ["latitude"],
                src_grid.lat.values,
                {"standard_name": "latitude"},
            ),
            "longitude": (
                ["longitude"],
                src_grid.lon.values,
                {"standard_name": "longitude"},
            ),
            "fixed_coord": ("fixed", [1, 2, 3]),
        },
    )

    # Regrid
    ds_regridded = regridder(ds)

    assert "temp" in ds_regridded.data_vars
    assert ds_regridded.temp.shape == (18, 36)
    assert "scalar" in ds_regridded.data_vars
    assert ds_regridded.scalar == 42.0
    assert "fixed_coord" in ds_regridded.coords


if __name__ == "__main__":
    pytest.main([__file__])
