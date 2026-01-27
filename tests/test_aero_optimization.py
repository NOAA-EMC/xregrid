import dask.array as da
import numpy as np
import pytest
import os
import xarray as xr
from xregrid import Regridder, create_global_grid
from xregrid.viz import plot_static


def test_no_hidden_compute_on_weight_load():
    """Verify that dask-backed coordinates are not computed when loading weights."""
    # We use a callback to detect compute
    compute_count = 0

    def count_compute(key, value, dsk):
        nonlocal compute_count
        compute_count += 1

    from dask.callbacks import Callback

    # Create dummy grid with dask coordinates
    lat = da.from_array(np.linspace(-90, 90, 10), chunks=5)
    lon = da.from_array(np.linspace(0, 360, 20), chunks=10)

    src_grid = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    # Create a weights file first (this WILL trigger compute because we generate weights)
    # We use a small grid that matches the mock in conftest
    filename = "test_weights_compute.nc"
    if os.path.exists(filename):
        os.remove(filename)

    _ = Regridder(src_grid, src_grid, filename=filename, reuse_weights=False)

    # Now try to load with reuse_weights=True and check if coordinates are computed
    class ComputeCounter(Callback):
        def _pretask(self, key, dsk, state):
            nonlocal compute_count
            compute_count += 1

    with ComputeCounter():
        compute_count = 0
        _ = Regridder(src_grid, src_grid, filename=filename, reuse_weights=True)

    # compute_count should be 0 for src_grid coordinates.
    # Note: Regridder._load_weights calls ds_weights.load(), which computes weight variables in that file,
    # but it should NOT compute src_grid coordinates from the input dataset.
    assert compute_count == 0

    if os.path.exists(filename):
        os.remove(filename)


def test_plot_static_robust_slicing():
    """Verify plot_static correctly handles non-standard dimension orders using cf-xarray."""
    # Create a 3D DataArray where spatial dims are NOT the last two
    # dims: ('lat', 'time', 'lon')
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20)
    time = [0, 1]

    data = np.random.rand(10, 2, 20)
    da_test = xr.DataArray(
        data,
        dims=("lat", "time", "lon"),
        coords={"lat": lat, "lon": lon, "time": time},
        name="test_data",
    )
    da_test.lat.attrs["standard_name"] = "latitude"
    da_test.lon.attrs["standard_name"] = "longitude"

    # This should slice 'time' (index 0) and plot ('lat', 'lon')
    # We check that the warning is issued and mentions 'time'
    with pytest.warns(
        UserWarning, match=r"Automatically selecting the first slice along \['time'\]"
    ):
        im = plot_static(da_test)

    assert im is not None


def test_weight_persistence_robustness():
    """Verify that weight attributes survive NetCDF round-trip as tuples."""
    src_grid = create_global_grid(10, 10)
    filename = "test_persistence_opt.nc"
    if os.path.exists(filename):
        os.remove(filename)

    regridder = Regridder(src_grid, src_grid, filename=filename, reuse_weights=True)

    # Check attributes are tuples
    assert isinstance(regridder._shape_source, tuple)
    assert isinstance(regridder._dims_source, tuple)

    # Re-load from same file
    regridder2 = Regridder(src_grid, src_grid, filename=filename, reuse_weights=True)
    assert regridder2._shape_source == regridder._shape_source
    assert regridder2._dims_source == regridder._dims_source
    assert isinstance(regridder2._shape_source, tuple)

    if os.path.exists(filename):
        os.remove(filename)
