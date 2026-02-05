import sys
import time
from unittest.mock import MagicMock

import dask.distributed
import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

# Mock ESMpy
mock_esmpy = MagicMock()
mock_esmpy.CoordSys.SPH_DEG = 1
mock_esmpy.StaggerLoc.CENTER = 0
mock_esmpy.StaggerLoc.CORNER = 1
mock_esmpy.GridItem.MASK = 1
mock_esmpy.RegridMethod.BILINEAR = 0
mock_esmpy.LogKind.MULTI = 1
sys.modules["esmpy"] = mock_esmpy

from xregrid.xregrid import _WORKER_CACHE, _apply_weights_core  # noqa: E402


def generate_mock_weights(n_src, n_dst, weights_per_row=4):
    nnz = n_dst * weights_per_row
    data = np.random.rand(nnz).astype(np.float32)
    row = np.repeat(np.arange(n_dst), weights_per_row)
    col = np.random.randint(0, n_src, size=nnz)
    return csr_matrix((data, (row, col)), shape=(n_dst, n_src))


def benchmark_dask(n_workers, n_chunks, n_lat=360, n_lon=720):
    cluster = dask.distributed.LocalCluster(
        n_workers=n_workers, threads_per_worker=1, processes=True
    )
    client = dask.distributed.Client(cluster)

    try:
        n_src = n_lat * n_lon
        n_dst = n_lat * n_lon
        weights = generate_mock_weights(n_src, n_dst)

        # 20 time steps
        data = np.random.rand(20, n_lat, n_lon).astype(np.float32)
        da = xr.DataArray(data, dims=("time", "lat", "lon")).chunk(
            {"time": 20 // n_chunks}
        )

        # We need to distribute weights to workers
        weights_key = "bench_weights"
        client.run(lambda: _WORKER_CACHE.update({weights_key: weights}))

        def run():
            out = xr.apply_ufunc(
                _apply_weights_core,
                da,
                kwargs={
                    "weights_matrix": weights_key,
                    "dims_source": ("lat", "lon"),
                    "shape_target": (n_lat, n_lon),
                    "skipna": False,
                },
                input_core_dims=[["lat", "lon"]],
                output_core_dims=[["lat_regridded", "lon_regridded"]],
                dask="parallelized",
                vectorize=False,
                output_dtypes=[da.dtype],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "lat_regridded": n_lat,
                        "lon_regridded": n_lon,
                    },
                    "allow_rechunk": True,
                },
            )
            return out.compute()

        # Warmup
        _ = run()

        start = time.perf_counter()
        _ = run()
        end = time.perf_counter()

        return end - start
    finally:
        client.close()
        cluster.close()


print("| Workers | Chunks | Resolution | Time | Speedup |")
print("|---------|--------|------------|------|--------|")
base_time = benchmark_dask(1, 4)
print(f"| 1 | 4 | 0.5° | {base_time:.2f}s | 1.0x |")

for w in [2, 4]:
    t = benchmark_dask(w, 4)
    print(f"| {w} | 4 | 0.5° | {t:.2f}s | {base_time / t:.1f}x |")
