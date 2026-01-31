import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_memory_opt_identity():
    """
    Verify that memory-optimized paths produce identical results to the previous implementation.
    Aero Protocol: Eager (NumPy) vs Lazy (Dask) identity.
    """
    res_src = 10.0
    res_tgt = 20.0
    src_grid = create_global_grid(res_src, res_src)
    tgt_grid = create_global_grid(res_tgt, res_tgt)

    ntime = 2
    nlat = src_grid.lat.size
    nlon = src_grid.lon.size

    # Create data with some NaNs
    data = np.random.rand(ntime, nlat, nlon).astype(np.float32)
    data[:, 0, 0] = np.nan

    da_src = xr.DataArray(
        data,
        coords={"time": np.arange(ntime), "lat": src_grid.lat, "lon": src_grid.lon},
        dims=("time", "lat", "lon"),
        name="test_data",
    )

    # 1. Eager path
    regridder = Regridder(src_grid, tgt_grid, method="bilinear", skipna=True)
    da_eager = regridder(da_src)

    # 2. Lazy path
    da_src_lazy = da_src.chunk({"time": 1})
    da_lazy = regridder(da_src_lazy)
    da_lazy_computed = da_lazy.compute()

    # Identity check
    xr.testing.assert_allclose(da_eager, da_lazy_computed)

    # Dtype preservation check
    assert da_eager.dtype == da_src.dtype
    assert da_lazy_computed.dtype == da_src.dtype


def test_repr_lazy_optimization():
    """Verify that __repr__ is lazy for large grids."""
    res_src = 10.0
    res_tgt = 20.0
    src_grid = create_global_grid(res_src, res_src)
    tgt_grid = create_global_grid(res_tgt, res_tgt)

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # For small grid, quality should be present
    repr_str = repr(regridder)
    assert "unmapped=" in repr_str
    assert "quality=deferred" not in repr_str

    # Manually mock a large target shape to trigger lazy repr
    regridder._shape_target = (1000, 1001)  # > 1,000,000 pixels
    repr_str_large = repr(regridder)
    assert "quality=deferred" in repr_str_large
    assert "unmapped=" not in repr_str_large


def test_matmul_backend_agnostic():
    """Basic test for _matmul helper."""
    from xregrid.xregrid import _matmul
    from scipy.sparse import csr_matrix

    matrix = csr_matrix([[1, 0], [0, 2]])
    data = np.array([[10, 20], [30, 40]])

    # (matrix @ data.T).T
    # matrix @ data.T = [[1, 0], [0, 2]] @ [[10, 30], [20, 40]] = [[10, 30], [40, 80]]
    # result = [[10, 40], [30, 80]]

    expected = np.array([[10, 40], [30, 80]])
    result = _matmul(matrix, data)

    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
