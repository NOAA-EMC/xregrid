import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_empty_input_robustness():
    """Verify that Regridder handles zero-sized input dimensions."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    # Create an empty DataArray along a non-spatial dimension
    data = np.zeros((0, 18, 36))
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"lat": src.lat, "lon": src.lon, "time": []},
    )

    regridder = Regridder(src, tgt)
    res = regridder(da)

    assert res.shape == (0, 36, 72)
    assert res.dtype == da.dtype


def test_all_nan_input_robustness():
    """Verify that Regridder handles all-NaN input arrays."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    data = np.full((18, 36), np.nan)
    da = xr.DataArray(
        data, dims=("lat", "lon"), coords={"lat": src.lat, "lon": src.lon}
    )

    regridder = Regridder(src, tgt, skipna=True)
    res = regridder(da)

    assert np.isnan(res.values).all()


def test_cache_clearing():
    """Verify Regridder cache clearing methods exist and don't crash."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt)

    # Test class method
    Regridder.clear_cache()

    # Test instance method
    regridder.clear_instance_cache()

    # Check that deletion doesn't crash (triggers __del__)
    del regridder


if __name__ == "__main__":
    pytest.main([__file__])
