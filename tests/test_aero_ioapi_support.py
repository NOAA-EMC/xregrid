import pytest
import xarray as xr
from xregrid.utils import create_grid_from_ioapi


def test_create_grid_from_ioapi_lcc():
    """Verify IOAPI grid generation for LCC projection (Eager and Lazy)."""
    metadata = {
        "GDTYP": 2,
        "P_ALP": 30.0,
        "P_BET": 60.0,
        "P_GAM": -97.0,
        "XCENT": -97.0,
        "YCENT": 40.0,
        "XORIG": -1000.0,
        "YORIG": -1000.0,
        "XCELL": 500.0,
        "YCELL": 500.0,
        "NCOLS": 4,
        "NROWS": 4,
    }

    # 1. Eager test
    ds_eager = create_grid_from_ioapi(metadata)

    assert "x" in ds_eager.coords
    assert "y" in ds_eager.coords
    assert "lat" in ds_eager.coords
    assert "lon" in ds_eager.coords
    assert ds_eager.sizes["x"] == 4
    assert ds_eager.sizes["y"] == 4
    assert ds_eager.attrs["ioapi_GDTYP"] == 2

    # 2. Lazy test
    ds_lazy = create_grid_from_ioapi(metadata, chunks={"x": 2, "y": 2})
    assert hasattr(ds_lazy.lat.data, "dask")

    ds_lazy_comp = ds_lazy.compute()
    xr.testing.assert_allclose(ds_eager, ds_lazy_comp)


def test_create_grid_from_ioapi_latlon():
    """Verify IOAPI grid generation for Lat-Lon."""
    metadata = {
        "GDTYP": 1,
        "P_ALP": 0.0,
        "P_BET": 0.0,
        "P_GAM": 0.0,
        "XCENT": 0.0,
        "YCENT": 0.0,
        "XORIG": -10.0,
        "YORIG": 40.0,
        "XCELL": 1.0,
        "YCELL": 1.0,
        "NCOLS": 10,
        "NROWS": 10,
    }
    ds = create_grid_from_ioapi(metadata)
    assert ds.sizes["x"] == 10
    assert ds.sizes["y"] == 10
    # For Lat-Lon, pyproj transform from EPSG:4326 to EPSG:4326 should be identity
    # but create_grid_from_crs might return lat/lon that are slightly different due to transform
    assert ds.lat.min() >= 40.0


if __name__ == "__main__":
    pytest.main([__file__])
