import sys
import time
from unittest.mock import MagicMock

import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

# Mock ESMpy before importing xregrid
mock_esmpy = MagicMock()
mock_esmpy.CoordSys.SPH_DEG = 1
mock_esmpy.StaggerLoc.CENTER = 0
mock_esmpy.StaggerLoc.CORNER = 1
mock_esmpy.GridItem.MASK = 1
mock_esmpy.RegridMethod.BILINEAR = 0
mock_esmpy.RegridMethod.CONSERVE = 1
mock_esmpy.RegridMethod.NEAREST_STOD = 2
mock_esmpy.RegridMethod.NEAREST_DTOS = 3
mock_esmpy.RegridMethod.PATCH = 4
mock_esmpy.UnmappedAction.IGNORE = 1
mock_esmpy.ExtrapMethod.NEAREST_STOD = 0
mock_esmpy.ExtrapMethod.NEAREST_IDAVG = 1
mock_esmpy.ExtrapMethod.CREEP_FILL = 2
mock_esmpy.LogKind.MULTI = 1
sys.modules["esmpy"] = mock_esmpy

from xregrid.xregrid import _WORKER_CACHE, _apply_weights_core  # noqa: E402


def generate_mock_weights(n_src, n_dst, weights_per_row=4):
    """Generate a mock CSR matrix for benchmarking."""
    nnz = n_dst * weights_per_row
    data = np.random.rand(nnz).astype(np.float32)
    row = np.repeat(np.arange(n_dst), weights_per_row)
    col = np.random.randint(0, n_src, size=nnz)
    return csr_matrix((data, (row, col)), shape=(n_dst, n_src))


def benchmark_apply(
    res_name, n_lat, n_lon, target_n_lat, target_n_lon, n_time=1, skipna=False
):
    n_src = n_lat * n_lon
    n_dst = target_n_lat * target_n_lon

    weights = generate_mock_weights(n_src, n_dst)
    data = np.random.rand(n_time, n_lat, n_lon).astype(np.float32)
    da = xr.DataArray(data, dims=("time", "lat", "lon"))
    if skipna:
        da.values[:, 0:10, 0:10] = np.nan  # Add some NaNs

    def run():
        return xr.apply_ufunc(
            _apply_weights_core,
            da,
            kwargs={
                "weights_matrix": weights,
                "dims_source": ("lat", "lon"),
                "shape_target": (target_n_lat, target_n_lon),
                "skipna": skipna,
            },
            input_core_dims=[["lat", "lon"]],
            output_core_dims=[["lat_regridded", "lon_regridded"]],
            dask="parallelized",
            vectorize=False,
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "lat_regridded": target_n_lat,
                    "lon_regridded": target_n_lon,
                },
                "allow_rechunk": True,
            },
        )

    # Warmup
    _ = run()

    start = time.perf_counter()
    n_runs = 5 if n_src < 2_000_000 else 2
    for _ in range(n_runs):
        _ = run()
    end = time.perf_counter()

    avg_time = (end - start) / n_runs
    return avg_time


def benchmark_stationary_mask(n_lat, n_lon, n_time=10):
    n_src = n_lat * n_lon
    n_dst = n_lat * n_lon  # Same size for simplicity

    weights = generate_mock_weights(n_src, n_dst)
    data = np.random.rand(n_time, n_lat, n_lon).astype(np.float32)
    data[:, 0:50, 0:50] = np.nan  # Stationary mask

    _WORKER_CACHE.clear()
    weights_key = "bench_weights"
    _WORKER_CACHE[weights_key] = weights

    # Time with skipna=True
    start = time.perf_counter()
    _ = _apply_weights_core(
        data, weights_key, ("lat", "lon"), (n_lat, n_lon), skipna=True
    )
    end = time.perf_counter()

    total_time = end - start
    return total_time / n_time


resolutions = [
    ("1.0°", 180, 360, 180, 360),
    ("0.5°", 360, 720, 360, 720),
    ("0.25°", 720, 1440, 720, 1440),
    ("0.1°", 1800, 3600, 1800, 3600),
]

print("## Single Time Step Performance (skipna=False)")
print("| Resolution | Grid Points | XRegrid Apply Time |")
print("|------------|-------------|--------------------|")
for name, ny, nx, tny, tnx in resolutions:
    t = benchmark_apply(name, ny, nx, tny, tnx)
    print(f"| {name} | {ny * nx:,} | {t * 1000:.2f} ms |")

print("\n## Multi-Time Step Performance (Stationary Mask Caching)")
print("| Time Steps | Resolution | Avg Time per Step |")
print("|------------|------------|-------------------|")
for n_t in [1, 10, 100]:
    for name, ny, nx, tny, tnx in [
        ("1.0°", 180, 360, 180, 360),
        ("0.25°", 720, 1440, 720, 1440),
    ]:
        t_per_step = benchmark_stationary_mask(ny, nx, n_time=n_t)
        print(f"| {n_t} | {name} | {t_per_step * 1000:.2f} ms |")
