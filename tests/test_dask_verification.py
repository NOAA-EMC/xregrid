
import pytest
import xarray as xr
import numpy as np
import dask.distributed
from xregrid import Regridder, create_global_grid
import scipy.sparse as sp
import time

def test_dask_parallel_regridding():
    """
    Test that running with parallel=True creates the same weights as serial execution.
    Also verifies lazy initialization.
    """
    # Create LocalCluster for testing
    cluster = dask.distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = dask.distributed.Client(cluster)

    try:
        source_grid = create_global_grid(10, 10)
        target_grid = create_global_grid(5, 5)

        # 1. Generate weights in serial
        regridder_serial = Regridder(source_grid, target_grid, method="bilinear", parallel=False)
        w_serial = regridder_serial._weights_matrix

        # 2a. Generate weights in parallel (Eager)
        print(f"Using Dask Client: {client}")
        regridder_eager = Regridder(source_grid, target_grid, method="bilinear", parallel=True, compute=True)
        w_eager = regridder_eager._weights_matrix

        # 3a. Compare Eager
        assert w_serial.shape == w_eager.shape
        assert w_serial.nnz == w_eager.nnz
        diff_eager = (w_serial - w_eager)
        assert np.abs(diff_eager.data).max() < 1e-10 if diff_eager.nnz > 0 else True

        # 2b. Generate weights in parallel (Lazy)
        regridder_lazy = Regridder(source_grid, target_grid, method="bilinear", parallel=True, compute=False)

        # Verify persist mechanism (should just return self)
        assert regridder_lazy.persist() is regridder_lazy

        # Verify it hasn't computed yet
        assert regridder_lazy._weights_matrix is None
        assert regridder_lazy._dask_futures is not None

        # Trigger compute
        print("Triggering compute on lazy regridder...")
        regridder_lazy.compute()
        w_lazy = regridder_lazy._weights_matrix

        assert w_lazy is not None
        assert regridder_lazy._dask_futures is None # should be cleared

        # 3b. Compare Lazy
        assert w_serial.shape == w_lazy.shape
        assert w_serial.nnz == w_lazy.nnz
        diff_lazy = (w_serial - w_lazy)
        assert np.abs(diff_lazy.data).max() < 1e-10 if diff_lazy.nnz > 0 else True

        print("Generic identity verification successful")

        # 4. Compare regridding result on dummy data
        data = np.random.rand(source_grid.sizes["lat"], source_grid.sizes["lon"])
        da = xr.DataArray(data, coords={"lat": source_grid.lat, "lon": source_grid.lon}, dims=["lat", "lon"])

        res_serial = regridder_serial(da)
        res_parallel = regridder_lazy(da) # Should be identical

        xr.testing.assert_allclose(res_serial, res_parallel)

        # 5. Test auto-compute on call
        regridder_auto = Regridder(source_grid, target_grid, method="bilinear", parallel=True, compute=False)
        assert regridder_auto._weights_matrix is None
        print("Triggering auto-compute via __call__...")
        res_auto = regridder_auto(da)
        assert regridder_auto._weights_matrix is not None
        xr.testing.assert_allclose(res_serial, res_auto)

        print("Dask parallel verification successful!")

    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    test_dask_parallel_regridding()
