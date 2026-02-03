"""
Larger-than-Memory Regridding with Dask
=======================================

This example demonstrates how XRegrid leverages the Dask backend to process
datasets that are larger than the available system memory. This is common
when working with high-resolution global models or long time series.

XRegrid uses `xarray.apply_ufunc` with `dask='parallelized'`, allowing it to
distribute the regridding operation across multiple Dask workers. Each worker
only processes a small chunk of data at a time, keeping the memory footprint low.

Key concepts demonstrated:
- Creating synthetic large-scale datasets with Dask.
- Parallelizing weight generation across workers.
- Lazy application of regridding weights.
- Memory-efficient processing on a local machine.
"""

import xarray as xr
import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster
import time
from xregrid import Regridder


def run_example():
    # 1. Setup a local Dask cluster
    # On a laptop, this allows us to use multiple CPU cores and manage memory.
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit="2GB")
    client = Client(cluster)
    print(f"Dask Dashboard: {client.dashboard_link}")

    # 2. Create a synthetic high-resolution dataset (larger than memory)
    # Global 0.1 degree grid: 3600 x 1800 points
    # 100 time steps
    # Total size: 100 * 3600 * 1800 * 8 bytes (float64) ≈ 5.18 GB
    # This dataset is larger than the 2GB limit we set for our workers,
    # demonstrating that we can process it chunk-by-chunk.

    nt, nlat, nlon = 100, 1800, 3600
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(-180, 180, nlon)

    # Create lazy Dask array for data
    # Chunked along time and space
    data = da.random.random((nt, nlat, nlon), chunks=(10, 450, 900))

    ds_src = xr.Dataset(
        {"temp": (["time", "lat", "lon"], data)},
        coords={
            "time": np.arange(nt),
            "lat": (["lat"], lats, {"units": "degrees_north"}),
            "lon": (["lon"], lons, {"units": "degrees_east"}),
        },
    )

    print(f"Source dataset size: {ds_src.nbytes / 1e9:.2f} GB")
    print(f"Number of chunks: {data.npartitions}")

    # 3. Define a coarser target grid (e.g., 1.0 degree)
    target_lats = np.arange(-90, 91, 1.0)
    target_lons = np.arange(-180, 180, 1.0)
    ds_tgt = xr.Dataset(
        coords={
            "lat": (["lat"], target_lats, {"units": "degrees_north"}),
            "lon": (["lon"], target_lons, {"units": "degrees_east"}),
        }
    )

    # 4. Initialize Regridder with parallel=True
    # Setting parallel=True tells XRegrid to use the Dask cluster to
    # calculate the regridding weights in parallel, which is much faster
    # for high-resolution grids.
    print("\nGenerating weights in parallel...")
    start = time.time()
    regridder = Regridder(
        ds_src, ds_tgt, method="bilinear", periodic=True, parallel=True
    )
    print(f"Weight generation took: {time.time() - start:.2f}s")

    # 5. Apply regridding (Lazy)
    # The result is another Dask-backed xarray DataArray.
    # No actual computation has happened yet!
    print("\nApplying regridding (lazy)...")
    temp_regridded = regridder(ds_src.temp)

    print(f"Is result lazy? {temp_regridded.chunks is not None}")
    print(f"Result shape: {temp_regridded.shape}")

    # 6. Trigger computation
    # Now we compute the mean over time. Dask will load chunks of the 5GB
    # dataset, regrid them, and aggregate the results, all while staying
    # under the memory limit.
    print("Triggering computation...")
    start = time.time()
    result = temp_regridded.mean(dim="time").compute()
    print(f"Computation took: {time.time() - start:.2f}s")

    print("\nRegridding successful!")
    print(f"Final mean of regridded data: {result.mean().values:.4f}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    # Wrap in try-except because esmpy might not be available in all environments
    try:
        run_example()
    except ImportError as e:
        print(f"Error: {e}")
        print("This example requires esmpy, dask, and distributed to be installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
