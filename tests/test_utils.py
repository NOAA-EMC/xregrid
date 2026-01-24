import xarray as xr
from xregrid import create_global_grid, create_regional_grid, load_esmf_file
import os


def test_create_global_grid():
    ds = create_global_grid(res_lat=10, res_lon=20)
    assert "lat" in ds
    assert "lon" in ds
    assert ds.lat.size == 18  # 180 / 10
    assert ds.lon.size == 18  # 360 / 20
    assert "lat_b" in ds
    assert "lon_b" in ds
    assert ds.lat_b.size == 19
    assert ds.lon_b.size == 19
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert "history" in ds.attrs


def test_create_regional_grid():
    ds = create_regional_grid(
        lat_range=(-45, 45), lon_range=(0, 90), res_lat=5, res_lon=5
    )
    assert ds.lat.size == 18  # 90 / 5
    assert ds.lon.size == 18  # 90 / 5
    assert ds.lat.min() == -42.5
    assert ds.lat.max() == 42.5
    assert "lat_b" in ds
    assert ds.lat_b.min() == -45
    assert ds.lat_b.max() == 45


def test_load_esmf_file(tmp_path):
    # Create a dummy NetCDF file
    filepath = os.path.join(tmp_path, "test_mesh.nc")
    ds_orig = xr.Dataset({"test": (("x",), [1, 2, 3])})
    ds_orig.to_netcdf(filepath)

    ds_loaded = load_esmf_file(filepath)
    assert "test" in ds_loaded
    assert "history" in ds_loaded.attrs
    assert "Loaded ESMF file" in ds_loaded.attrs["history"]
