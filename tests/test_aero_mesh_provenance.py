import numpy as np
import pytest
import xarray as xr
from xregrid.utils import create_mesh_from_coords

try:
    import dask.array as da
except ImportError:
    da = None


def test_create_mesh_from_coords_aero():
    """
    Double-Check Test for create_mesh_from_coords.
    Verifies Eager (NumPy) and Lazy (Dask) backends yield identical results
    and maintain scientific provenance.
    """
    # 1. Setup sample coordinates (Lambert Conformal-ish)
    x = np.linspace(-1000, 1000, 10)
    y = np.linspace(-1000, 1000, 10)
    crs = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    # 2. Eager execution
    ds_eager = create_mesh_from_coords(x, y, crs)

    # Assertions for Eager
    assert isinstance(ds_eager, xr.Dataset)
    assert "lat" in ds_eager
    assert "lon" in ds_eager
    assert "x" in ds_eager
    assert "y" in ds_eager
    assert ds_eager.attrs["grid_mapping"] == "spatial_ref"
    assert "spatial_ref" in ds_eager
    assert "Eager" in ds_eager.attrs["history"]
    assert "Extent:" in ds_eager.attrs["history"]

    # Check that it's actually NumPy-backed
    assert not hasattr(ds_eager.lat.data, "dask")

    # 3. Lazy execution
    if da is None:
        pytest.skip("Dask not installed, skipping lazy check.")

    x_lazy = da.from_array(x, chunks=5)
    y_lazy = da.from_array(y, chunks=5)

    ds_lazy = create_mesh_from_coords(x_lazy, y_lazy, crs)

    # Assertions for Lazy
    assert "Lazy" in ds_lazy.attrs["history"]
    # Lazy path should NOT have extent in history to avoid compute()
    assert "Extent:" not in ds_lazy.attrs["history"]
    assert hasattr(ds_lazy.lat.data, "dask")

    # 4. Numerical Verification (The "Double-Check")
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # 5. Verify Metadata propagation
    assert ds_eager.lat.attrs["units"] == "degrees_north"
    assert ds_eager.x.attrs["standard_name"] == "projection_x_coordinate"
    assert ds_eager.x.attrs["grid_mapping"] == "spatial_ref"


def test_create_mesh_from_coords_regression_fix():
    """
    Verify the fix for the dimension mismatch regression and conditional metadata.
    """
    # 1. Test DataArray inputs with different dimension names
    x_da = xr.DataArray(np.linspace(0, 10, 5), dims=["lon"], name="my_lon")
    y_da = xr.DataArray(np.linspace(0, 10, 5), dims=["lat"], name="my_lat")
    crs_proj = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    ds = create_mesh_from_coords(x_da, y_da, crs_proj)

    # Should have 5 points, not 25 (if it had broadcasted incorrectly)
    assert ds.sizes["n_pts"] == 5
    assert ds.x.attrs["standard_name"] == "projection_x_coordinate"

    # 2. Test geographic CRS metadata
    crs_geo = "EPSG:4326"
    ds_geo = create_mesh_from_coords(x_da, y_da, crs_geo)

    assert ds_geo.x.attrs["standard_name"] == "longitude"
    assert ds_geo.x.attrs["units"] == "degrees_east"
    assert ds_geo.y.attrs["standard_name"] == "latitude"
    assert ds_geo.y.attrs["units"] == "degrees_north"


if __name__ == "__main__":
    test_create_mesh_from_coords_aero()
