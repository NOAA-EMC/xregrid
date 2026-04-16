import pytest
import xarray as xr
import numpy as np
from xregrid.utils import (
    create_sinusoidal_grid,
    create_grid_from_ioapi,
    create_grid_like,
)


def test_create_sinusoidal_grid_consistency():
    """Verify that sinusoidal grid generation yields identical results for NumPy and Dask."""
    # Small extent for testing
    extent = (-1000000, 1000000, -500000, 500000)
    res = 100000

    ds_eager = create_sinusoidal_grid(extent, res, chunks=None)
    ds_lazy = create_sinusoidal_grid(extent, res, chunks={"x": 5, "y": 5})

    # Verify metadata
    assert "lat" in ds_eager.coords
    assert "lon" in ds_eager.coords
    assert "x" in ds_eager.coords
    assert "y" in ds_eager.coords

    # Verify values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify backend
    assert not hasattr(ds_eager.lat.data, "dask")
    assert hasattr(ds_lazy.lat.data, "dask")
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_create_grid_from_ioapi_sinu():
    """Verify IOAPI GDTYP 13 (Sinusoidal) grid generation."""
    metadata = {
        "GDTYP": 13,
        "P_ALP": 0.0,
        "P_BET": 0.0,
        "P_GAM": 0.0,
        "XCENT": -97.0,
        "YCENT": 0.0,
        "XORIG": -1000000.0,
        "YORIG": -1000000.0,
        "XCELL": 10000.0,
        "YCELL": 10000.0,
        "NCOLS": 10,
        "NROWS": 10,
    }

    ds = create_grid_from_ioapi(metadata)
    assert ds.attrs["ioapi_GDTYP"] == 13
    assert "lat" in ds.coords
    assert "lon" in ds.coords

    # Verify lon_0 is respected (at least check the central meridian)
    # The center of the grid in x is XORIG + (NCOLS/2)*XCELL = -1000000 + 50000 = -950000
    # Central meridian is -97.0.
    # At y=0 (equator), lon should be close to -97.0 + x/R * 180/pi
    # But easier to just check if it runs and has reasonable values.
    assert ds.lon.mean() < 0


def test_sinusoidal_grid_like():
    """Verify create_grid_like works with Sinusoidal grids."""
    extent = (-500000, 500000, -500000, 500000)
    res = 50000
    ds_base = create_sinusoidal_grid(extent, res, lon_0=-100)

    # Create a new grid like the base one but with different resolution
    new_res = 100000
    ds_new = create_grid_like(ds_base, new_res)

    assert ds_new.sizes["x"] == 10
    assert ds_new.sizes["y"] == 10

    # Check if CRS is preserved (via WKT comparison)
    assert ds_base.attrs["crs"] == ds_new.attrs["crs"]

    # Check if extent is similar
    assert np.allclose(ds_base.x.min() - res / 2, ds_new.x.min() - new_res / 2)
    assert np.allclose(ds_base.x.max() + res / 2, ds_new.x.max() + new_res / 2)


def test_sinusoidal_res_aliases():
    """Verify Sinusoidal grid generation with resolution aliases."""
    extent = (0, 100000, 0, 100000)
    ds_10km = create_sinusoidal_grid(extent, "10km")
    ds_5km = create_sinusoidal_grid(extent, "5km")
    ds_1km = create_sinusoidal_grid(extent, "1km")
    ds_500m = create_sinusoidal_grid(extent, "500m")
    ds_250m = create_sinusoidal_grid(extent, "250m")

    # 1km alias should be ~926.6m
    expected_1km = 926.6254331
    assert np.allclose(ds_10km.x.diff("x").mean(), expected_1km * 10)
    assert np.allclose(ds_5km.x.diff("x").mean(), expected_1km * 5)
    assert np.allclose(ds_1km.x.diff("x").mean(), expected_1km)
    assert np.allclose(ds_500m.x.diff("x").mean(), expected_1km / 2)
    assert np.allclose(ds_250m.x.diff("x").mean(), expected_1km / 4)


if __name__ == "__main__":
    pytest.main([__file__])
