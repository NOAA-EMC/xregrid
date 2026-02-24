import numpy as np
import pytest
import xarray as xr
import dask.array as da
from xregrid import Regridder, create_global_grid
from xregrid.utils import _find_coord


def test_enhanced_coord_discovery():
    """Verify enhanced coordinate discovery for non-standard names."""
    # Create dataset with 'xc' and 'yc' without CF attributes to force fallback
    # Use auxiliary coordinates to test lazy discovery
    ds = xr.Dataset(
        data_vars={"data": (["y", "x"], np.random.rand(10, 20))},
        coords={
            "yc": (["y", "x"], np.random.rand(10, 20)),
            "xc": (["y", "x"], np.random.rand(10, 20)),
        },
    )

    # 1. Eager
    lat = _find_coord(ds, "latitude")
    lon = _find_coord(ds, "longitude")

    assert lat.name == "yc"
    assert lon.name == "xc"

    # 2. Lazy
    ds_lazy = ds.chunk({"y": 5, "x": 10})
    lat_lazy = _find_coord(ds_lazy, "latitude")
    lon_lazy = _find_coord(ds_lazy, "longitude")

    assert lat_lazy.name == "yc"
    assert lon_lazy.name == "xc"
    assert hasattr(lat_lazy.data, "dask")


def test_auto_periodicity_detection():
    """Verify auto-periodicity detection logic."""
    # Global grid should be detected as periodic
    ds_global = create_global_grid(10, 10)
    regridder = Regridder(ds_global, ds_global, periodic=None)
    assert regridder.periodic is True

    # Regional grid should NOT be detected as periodic
    ds_regional = xr.Dataset(
        coords={
            "lat": (["lat"], np.linspace(20, 50, 10)),
            "lon": (["lon"], np.linspace(-100, -70, 20)),
        }
    )
    regridder_reg = Regridder(ds_regional, ds_regional, periodic=None)
    assert regridder_reg.periodic is False


def test_auto_periodicity_lazy():
    """Verify auto-periodicity detection handles lazy coordinates without compute."""
    # Test 1: 1D lazy coordinates (detection should stay False if no metadata)
    lat = da.linspace(-90, 90, 10, chunks=5)
    lon = da.linspace(0, 360, 21, chunks=10)[:-1]

    ds_lazy = xr.Dataset(
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        }
    )

    # Regridder should NOT compute the dask arrays for detection
    regridder = Regridder(ds_lazy, ds_lazy, periodic=None)
    assert regridder.periodic is False

    # Now add metadata
    ds_lazy.lon.attrs["boundary"] = "periodic"
    regridder_meta = Regridder(ds_lazy, ds_lazy, periodic=None)
    assert regridder_meta.periodic is True


def test_plot_comparison_dispatch(mocker):
    """Verify plot_comparison method correctly dispatches to viz."""
    # Mock viz functions
    mock_static = mocker.patch("xregrid.viz.plot_comparison")
    mock_interactive = mocker.patch("xregrid.viz.plot_comparison_interactive")

    src = create_global_grid(30, 30)
    regridder = Regridder(src, src, periodic=False)

    da_src = xr.DataArray(np.random.rand(6, 12), dims=("lat", "lon"))
    da_tgt = xr.DataArray(np.random.rand(6, 12), dims=("lat", "lon"))

    # Track A (Static)
    regridder.plot_comparison(da_src, da_tgt, mode="static", custom_kw="test")
    mock_static.assert_called_once_with(
        da_src, da_tgt, regridder=regridder, custom_kw="test"
    )

    # Track B (Interactive)
    regridder.plot_comparison(da_src, da_tgt, mode="interactive", rasterize=False)
    mock_interactive.assert_called_once_with(
        da_src, da_tgt, regridder=regridder, rasterize=False
    )


if __name__ == "__main__":
    pytest.main([__file__])
