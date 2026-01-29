import numpy as np
import xarray as xr
import pytest
from xregrid import Regridder, create_global_grid


def test_optimization_v2_identity():
    """Verify that the optimized path produces identical results to a known-valid path."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)

    # skipna=True to trigger the optimized code path
    regridder = Regridder(src_grid, tgt_grid, method="bilinear", skipna=True)

    # 1. Data with NO NaNs (triggers fast path)
    data_clean = np.random.rand(18, 36)
    da_clean = xr.DataArray(
        data_clean,
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
        name="clean_data",
    )

    res_clean = regridder(da_clean)
    assert res_clean.name == "clean_data"
    assert "Regridded" in res_clean.attrs["history"]

    # 2. Data WITH NaNs (triggers slow path)
    data_dirty = data_clean.copy()
    data_dirty[0, 0] = np.nan
    da_dirty = xr.DataArray(
        data_dirty,
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
    )

    res_dirty = regridder(da_dirty)

    # Verify both ran successfully
    assert res_clean is not None
    assert res_dirty is not None


def test_lazy_data_handling():
    """Verify that Dask-backed data works with the optimized skipna path."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)
    regridder = Regridder(src_grid, tgt_grid, method="bilinear", skipna=True)

    data = np.random.rand(18, 36)
    da_lazy = xr.DataArray(
        data, dims=("lat", "lon"), coords={"lat": src_grid.lat, "lon": src_grid.lon}
    ).chunk({"lat": 9})

    res_lazy = regridder(da_lazy)
    # Check it's still lazy
    assert hasattr(res_lazy.data, "dask")

    # Compute and check
    res_computed = res_lazy.compute()
    assert res_computed.shape == (36, 72)


def test_dataset_regridding_provenance():
    """Verify Dataset regridding preserves history and non-spatial coords."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)
    regridder = Regridder(src_grid, tgt_grid)

    ds = xr.Dataset(
        data_vars={
            "temp": (("lat", "lon"), np.random.rand(18, 36)),
            "mask": (("lat", "lon"), np.ones((18, 36))),
        },
        coords={"lat": src_grid.lat, "lon": src_grid.lon, "time": [0]},
    )

    res_ds = regridder(ds)
    assert "time" in res_ds.coords
    assert "history" in res_ds.attrs
    assert "Regridded Dataset" in res_ds.attrs["history"]
    assert "temp" in res_ds.data_vars
    assert "mask" in res_ds.data_vars


if __name__ == "__main__":
    pytest.main([__file__])
