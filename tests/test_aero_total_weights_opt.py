import numpy as np
import xarray as xr
from xregrid import Regridder
import xregrid.xregrid as xregrid_mod


def test_total_weights_distribution_eager_vs_lazy():
    """
    Aero Protocol: Verify that total_weights distribution works correctly
    and produces identical results for Eager (NumPy) and Lazy (Dask) paths
    when skipna=True.
    """
    from dask.distributed import Client, LocalCluster

    # Clear cache to ensure fresh test
    xregrid_mod._WORKER_CACHE.clear()

    # Create simple source and target grids
    ds_src = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(-90, 90, 10)),
            "lon": (["lon"], np.linspace(0, 360, 20)),
        }
    )
    # Ensure they have CF attributes for _get_mesh_info
    ds_src.lat.attrs["units"] = "degrees_north"
    ds_src.lon.attrs["units"] = "degrees_east"

    ds_tgt = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(-90, 90, 15)),
            "lon": (["lon"], np.linspace(0, 360, 25)),
        }
    )
    ds_tgt.lat.attrs["units"] = "degrees_north"
    ds_tgt.lon.attrs["units"] = "degrees_east"

    # Create source data with some NaNs
    data = np.random.rand(10, 20).astype(np.float32)
    data[0, 0] = np.nan
    da_src_numpy = xr.DataArray(
        data, coords=ds_src.coords, dims=("lat", "lon"), name="test"
    )

    # 1. Eager Path
    regridder = Regridder(ds_src, ds_tgt, method="bilinear", skipna=True)
    res_numpy = regridder(da_src_numpy)

    # 2. Lazy Path
    with LocalCluster(n_workers=1, threads_per_worker=1, processes=False) as cluster:
        with Client(cluster):
            da_src_dask = da_src_numpy.chunk({"lat": 5, "lon": 10})
            res_dask = regridder(da_src_dask)

            # Verify identity
            xr.testing.assert_allclose(res_numpy, res_dask.compute())

    # Verify that history contains the new metadata
    assert "ESMF/esmpy=" in res_numpy.attrs["history"]
    assert "skipna=True" in res_numpy.attrs["history"]
    assert "na_thres=1.0" in res_numpy.attrs["history"]

    # Check if the total weights key was created in _WORKER_CACHE
    tw_keys = [k for k in xregrid_mod._WORKER_CACHE.keys() if k.startswith("tw_")]
    assert len(tw_keys) > 0, "Total weights should have been cached with a key"

    # Check if weights_matrix was also cached
    w_keys = [k for k in xregrid_mod._WORKER_CACHE.keys() if k.startswith("weights_")]
    assert len(w_keys) > 0, "Weights matrix should have been cached with a key"


def test_provenance_with_extrap():
    """Verify that extrapolation metadata is included in history."""
    ds_src = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(-90, 90, 10)),
            "lon": (["lon"], np.linspace(0, 360, 20)),
        }
    )
    ds_src.lat.attrs["units"] = "degrees_north"
    ds_src.lon.attrs["units"] = "degrees_east"

    ds_tgt = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(-90, 90, 15)),
            "lon": (["lon"], np.linspace(0, 360, 25)),
        }
    )
    ds_tgt.lat.attrs["units"] = "degrees_north"
    ds_tgt.lon.attrs["units"] = "degrees_east"

    da_src = xr.DataArray(
        np.ones((10, 20)), coords=ds_src.coords, dims=("lat", "lon"), name="test"
    )

    regridder = Regridder(
        ds_src, ds_tgt, method="bilinear", extrap_method="nearest_s2d"
    )
    res = regridder(da_src)

    assert "extrap_method=nearest_s2d" in res.attrs["history"]
