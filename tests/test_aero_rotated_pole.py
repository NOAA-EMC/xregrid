import pytest
import xarray as xr
from xregrid import create_rotated_latlon_grid


def test_rotated_latlon_grid_eager_lazy():
    """
    Double-Check Test: Verify create_rotated_latlon_grid yields identical results
    for Eager (NumPy) and Lazy (Dask) backends and maintains CF compliance.
    """
    extent = (-5.0, 5.0, -5.0, 5.0)
    res = 1.0
    pole_lat = 37.5
    pole_lon = 177.5

    # 1. Eager Execution (NumPy)
    ds_eager = create_rotated_latlon_grid(
        extent=extent,
        res=res,
        grid_north_pole_lat=pole_lat,
        grid_north_pole_lon=pole_lon,
        add_bounds=True,
        chunks=None,
    )

    # 2. Lazy Execution (Dask)
    ds_lazy = create_rotated_latlon_grid(
        extent=extent,
        res=res,
        grid_north_pole_lat=pole_lat,
        grid_north_pole_lon=pole_lon,
        add_bounds=True,
        chunks={"rlat": 5, "rlon": 5},
    )

    # Verification of Laziness
    assert hasattr(ds_lazy.lat.data, "dask")
    assert hasattr(ds_lazy.lon.data, "dask")
    assert hasattr(ds_lazy.rlat_b.data, "dask")

    # Verification of Numerical Identity
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verification of CF Metadata
    assert ds_eager.attrs["grid_mapping"] == "rotated_pole"
    assert "rotated_pole" in ds_eager.data_vars
    assert (
        ds_eager.rotated_pole.attrs["grid_mapping_name"] == "rotated_latitude_longitude"
    )
    assert ds_eager.rlat.attrs["standard_name"] == "grid_latitude"
    assert ds_eager.rlon.attrs["standard_name"] == "grid_longitude"

    # Verification of Provenance
    assert "Created Rotated Lat-Lon grid" in ds_eager.attrs["history"]
    assert "Eager" in ds_eager.attrs["history"]
    assert "Lazy" in ds_lazy.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
