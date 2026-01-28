import numpy as np
import xarray as xr
import dask.array as da
from xregrid import Regridder, create_global_grid
from xregrid.viz import plot_comparison


def test_auto_bounds_conservative_numpy_dask():
    """Verify auto-bounds generation for conservative regridding on both NumPy and Dask."""
    # Create a grid WITHOUT bounds but with standard names
    lat = np.linspace(-85, 85, 10)
    lon = np.linspace(0, 350, 20)
    ds_src = xr.Dataset(coords={"lat": lat, "lon": lon})
    ds_src.lat.attrs["standard_name"] = "latitude"
    ds_src.lat.attrs["units"] = "degrees_north"
    ds_src.lon.attrs["standard_name"] = "longitude"
    ds_src.lon.attrs["units"] = "degrees_east"

    # Target grid with bounds
    ds_tgt = create_global_grid(20, 20)

    # 1. Eager path
    regridder_eager = Regridder(ds_src, ds_tgt, method="conservative")
    da_src_eager = xr.DataArray(
        np.random.rand(10, 20), dims=("lat", "lon"), coords=ds_src.coords
    )
    res_eager = regridder_eager(da_src_eager)

    # 2. Lazy path
    da_src_lazy = da_src_eager.chunk({"lat": 5, "lon": 10})
    res_lazy = regridder_eager(da_src_lazy)

    assert isinstance(res_lazy.data, da.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Automatically generated" in ds_src.attrs["history"]


def test_plot_comparison_smoke():
    """Smoke test for plot_comparison utility."""
    ds = create_global_grid(30, 30)
    da_coords = {c: ds.coords[c] for c in ["lat", "lon"]}
    da = xr.DataArray(np.random.rand(6, 12), dims=("lat", "lon"), coords=da_coords)

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")  # Non-interactive

    fig = plot_comparison(da, da)
    assert fig is not None
    plt.close(fig)
