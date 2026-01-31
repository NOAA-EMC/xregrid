import numpy as np
import xarray as xr
from xregrid import Regridder
from xregrid.utils import create_global_grid


def test_regridder_eager_vs_lazy_skipna():
    """
    Verify that Regridder produces identical results for Eager (NumPy) and Lazy (Dask)
    data, specifically testing the optimized skipna path.
    """
    # Create small source and target grids
    # Global grid 10x10 -> 18x36 pixels
    ds_src = create_global_grid(10, 10)
    ds_tgt = create_global_grid(15, 15)  # 12x24 pixels

    # Create data with some NaNs
    data = np.random.rand(18, 36).astype(np.float32)
    data[0, 0] = np.nan
    data[5, 5] = np.nan

    da_eager = xr.DataArray(
        data,
        coords={
            "lat": ds_src.coords["lat"],
            "lon": ds_src.coords["lon"],
        },
        dims=("lat", "lon"),
        name="test_data",
    )

    # Initialize Regridder
    # Note: In environments where ESMF is mocked, weights will be synthetic but consistent
    regridder = Regridder(ds_src, ds_tgt, method="bilinear", skipna=True)

    # 1. Eager application
    res_eager = regridder(da_eager)

    # 2. Lazy application
    da_lazy = da_eager.chunk({"lat": 9, "lon": 9})
    res_lazy = regridder(da_lazy).compute()

    # Verify results are identical
    xr.testing.assert_allclose(res_eager, res_lazy)

    # Basic sanity check on the output
    assert isinstance(res_eager, xr.DataArray)
    assert res_eager.shape == (12, 24)


def test_regridder_dataset_eager_vs_lazy():
    """Verify Dataset regridding for both backends (Aero Protocol)."""
    ds_src = create_global_grid(20, 20)
    ds_tgt = create_global_grid(30, 30)

    ds_src["var1"] = (("lat", "lon"), np.random.rand(9, 18))
    ds_src["var2"] = (("lat", "lon"), np.random.rand(9, 18))

    regridder = Regridder(ds_src, ds_tgt, method="bilinear")

    # Eager
    res_eager = regridder(ds_src)

    # Lazy
    ds_lazy = ds_src.chunk({"lat": 3, "lon": 3})
    res_lazy = regridder(ds_lazy).compute()

    xr.testing.assert_allclose(res_eager, res_lazy)
    assert "var1" in res_eager
    assert "var2" in res_eager
    assert res_eager["var1"].shape == (6, 12)


def test_total_weights_optimization():
    """Verify that the optimized total_weights calculation matches the old logic."""
    from scipy.sparse import csr_matrix

    # Create a synthetic sparse matrix
    data = np.array([0.5, 0.5, 1.0, 0.2, 0.8])
    row = np.array([0, 0, 1, 2, 2])
    col = np.array([0, 1, 1, 2, 3])
    weights_matrix = csr_matrix((data, (row, col)), shape=(3, 4))

    # Old logic
    n_src = 4
    total_weights_old = np.ones((1, n_src)) @ weights_matrix.T
    total_weights_old = np.asarray(total_weights_old).flatten()

    # New optimized logic
    total_weights_new = np.asarray(weights_matrix.sum(axis=1)).flatten()

    np.testing.assert_allclose(total_weights_old, total_weights_new)
