import numpy as np
import pytest
import xarray as xr
import dask.array as da
from xregrid import Regridder, create_global_grid


def test_auxiliary_coordinate_preservation():
    """
    Verify that auxiliary spatial coordinates are preserved and regridded.
    Follows Aero Protocol: Eager (NumPy) and Lazy (Dask) verification.
    """
    # Create grids
    src_grid = create_global_grid(10, 10)  # 18x36
    tgt_grid = create_global_grid(5, 5)  # 36x72

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # 1. Eager (NumPy) DataArray Test
    data = np.random.rand(18, 36)
    alt = np.random.rand(18, 36)

    da_eager = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={
            "lat": src_grid.lat,
            "lon": src_grid.lon,
            "altitude": (("lat", "lon"), alt),
        },
        name="test_data",
    )

    res_eager = regridder(da_eager)

    assert "altitude" in res_eager.coords
    assert res_eager.altitude.shape == (36, 72)
    assert res_eager.shape == (36, 72)

    # 2. Lazy (Dask) DataArray Test
    da_lazy = da_eager.chunk({"lat": 9, "lon": 18})
    res_lazy = regridder(da_lazy)

    assert isinstance(res_lazy.data, da.Array)
    assert isinstance(res_lazy.altitude.data, da.Array)

    res_lazy_comp = res_lazy.compute()

    xr.testing.assert_allclose(res_eager, res_lazy_comp)
    xr.testing.assert_allclose(res_eager.altitude, res_lazy_comp.altitude)


def test_auxiliary_coordinate_preservation_dataset():
    """Verify auxiliary coordinates are preserved in Datasets."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    ds = xr.Dataset(
        data_vars={
            "temp": (("lat", "lon"), np.random.rand(18, 36)),
        },
        coords={
            "lat": src_grid.lat,
            "lon": src_grid.lon,
            "sensor_angle": (("lat", "lon"), np.random.rand(18, 36)),
            "static_metadata": "fixed_value",
        },
    )

    res_ds = regridder(ds)

    assert "sensor_angle" in res_ds.coords
    assert res_ds.sensor_angle.shape == (36, 72)
    assert "static_metadata" in res_ds.coords
    assert res_ds.static_metadata == "fixed_value"
    assert res_ds.temp.shape == (36, 72)


def test_mutual_auxiliary_coordinate_recursion():
    """Verify that mutual dependencies between coordinates don't cause infinite recursion."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)
    regridder = Regridder(src_grid, tgt_grid)

    # Create mutual auxiliary coordinates
    lon_aux = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
        name="lon_aux",
    )
    lat_aux = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
        name="lat_aux",
    )

    # Link them
    lon_aux = lon_aux.assign_coords(lat_aux=lat_aux)
    lat_aux = lat_aux.assign_coords(lon_aux=lon_aux)

    da = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        coords={
            "lat": src_grid.lat,
            "lon": src_grid.lon,
            "lat_aux": lat_aux,
            "lon_aux": lon_aux,
        },
        name="test_data",
    )

    # This should not raise RecursionError
    res = regridder(da)
    assert "lat_aux" in res.coords
    assert "lon_aux" in res.coords
    assert res.lat_aux.shape == (36, 72)


if __name__ == "__main__":
    pytest.main([__file__])
