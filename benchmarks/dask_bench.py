import time
import numpy as np
import xarray as xr
import xesmf as xe
from xregrid import ESMPyRegridder
import os
import gc
import dask.array as da

def benchmark_dask(name, nlat, nlon, ntime):
    print(f"\n--- Benchmarking DASK {name} ({ntime} chunks) ---")

    # Simple arrays
    data_in = da.random.random((ntime, nlat, nlon), chunks=(1, nlat, nlon)).astype(np.float32)
    lat = np.linspace(-89.9, 89.9, nlat)
    lon = np.linspace(0, 359.5, nlon)

    da_in = xr.DataArray(data_in, dims=("time", "lat", "lon"), coords={"lat": lat, "lon": lon})

    grid_ds = xr.Dataset(coords={"lat": (["lat"], lat), "lon": (["lon"], lon)})

    regridder_xesmf = xe.Regridder(grid_ds, grid_ds, method="bilinear", periodic=True)
    regridder_xregrid = ESMPyRegridder(grid_ds, grid_ds, method="bilinear", periodic=True)

    # Warmup
    _ = regridder_xesmf(da_in.isel(time=slice(0, 1))).compute()
    _ = regridder_xregrid(da_in.isel(time=slice(0, 1))).compute()

    # xESMF
    start = time.perf_counter()
    res_xesmf = regridder_xesmf(da_in)
    _ = res_xesmf.compute()
    t_xesmf = time.perf_counter() - start

    # XRegrid
    start = time.perf_counter()
    res_xregrid = regridder_xregrid(da_in)
    _ = res_xregrid.compute()
    t_xregrid = time.perf_counter() - start

    print(f"xESMF: {t_xesmf:.6f}s, XRegrid: {t_xregrid:.6f}s")
    print(f"Speedup: {t_xesmf / t_xregrid:.1f}x")

benchmark_dask("1.0° Global", 180, 360, 20)
