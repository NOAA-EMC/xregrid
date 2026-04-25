import xarray as xr
import pytest
from xregrid.utils import create_lcc_grid


def test_lcc_grid_backend_consistency():
    """Verify that LCC grid generation yields identical results for NumPy and Dask."""
    pytest.importorskip("pyproj")

    # Define a small LCC grid
    extent = (-100000, 100000, -100000, 100000)
    res = 20000
    lat_1, lat_2 = 33, 45
    lat_0, lon_0 = 40, -97

    ds_eager = create_lcc_grid(
        extent=extent,
        res=res,
        lat_1=lat_1,
        lat_2=lat_2,
        lat_0=lat_0,
        lon_0=lon_0,
        chunks=None,
    )

    ds_lazy = create_lcc_grid(
        extent=extent,
        res=res,
        lat_1=lat_1,
        lat_2=lat_2,
        lat_0=lat_0,
        lon_0=lon_0,
        chunks={"x": 5, "y": 5},
    )

    # 1. Numerical Consistency
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # 2. Laziness Check
    # In xarray, dimension coordinates are eager.
    # Non-dimension coordinates and DataArrays should be lazy.
    assert hasattr(ds_lazy.lat.data, "dask")
    assert hasattr(ds_lazy.lon.data, "dask")
    assert hasattr(ds_lazy.lat_b.data, "dask")
    assert hasattr(ds_lazy.lon_b.data, "dask")

    # 3. Metadata and CF-compliance
    assert "lat" in ds_eager.coords
    assert "lon" in ds_eager.coords
    assert ds_eager.lat.attrs["standard_name"] == "latitude"
    assert ds_eager.lon.attrs["standard_name"] == "longitude"
    assert "crs" in ds_eager.attrs
    assert "Lambert Conic Conformal" in ds_eager.attrs["crs"]

    # 4. Provenance
    assert "history" in ds_eager.attrs
    assert "Created Lambert Conformal Conic grid" in ds_eager.attrs["history"]
    assert "(Eager)" in ds_eager.attrs["history"]
    assert "(Lazy)" in ds_lazy.attrs["history"]


def test_lcc_grid_resolution_tuple():
    """Verify LCC grid works with a tuple for resolution."""
    pytest.importorskip("pyproj")
    extent = (-10000, 10000, -10000, 10000)
    res = (1000, 2000)

    ds = create_lcc_grid(
        extent=extent,
        res=res,
        lat_1=30,
        lat_2=60,
        lat_0=45,
        lon_0=-100,
    )

    assert ds.x.size == 20  # 20000 / 1000
    assert ds.y.size == 10  # 20000 / 2000
