import time
import numpy as np
import xarray as xr
import xesmf as xe
from xregrid import ESMPyRegridder
import os
import gc

def create_sample_dataset(nlat, nlon):
    """Create a sample dataset with synthetic data."""
    lat = np.linspace(-89.9, 89.9, nlat)
    lon = np.linspace(0, 359.5, nlon)
    data = np.random.rand(nlat, nlon).astype(np.float32)
    ds = xr.Dataset(
        {"temperature": (["lat", "lon"], data)},
        coords={"lat": (["lat"], lat), "lon": (["lon"], lon)},
    )
    return ds

def benchmark_one(name, nlat, nlon):
    print(f"\n--- Benchmarking {name} ---")
    source_ds = create_sample_dataset(nlat, nlon)
    target_ds = create_sample_dataset(nlat, nlon)

    regridder_xesmf = xe.Regridder(source_ds, target_ds, method="bilinear", periodic=True)
    regridder_xregrid = ESMPyRegridder(source_ds, target_ds, method="bilinear", periodic=True)

    data_in = source_ds["temperature"]

    # Warmup
    _ = regridder_xesmf(data_in)
    _ = regridder_xregrid(data_in)

    # xESMF
    start = time.perf_counter()
    _ = regridder_xesmf(data_in)
    t_xesmf = time.perf_counter() - start

    # XRegrid
    start = time.perf_counter()
    _ = regridder_xregrid(data_in)
    t_xregrid = time.perf_counter() - start

    print(f"xESMF: {t_xesmf:.6f}s, XRegrid: {t_xregrid:.6f}s")
    print(f"Speedup: {t_xesmf / t_xregrid:.1f}x")

    # Clean up to save memory
    del regridder_xesmf
    del regridder_xregrid
    del source_ds
    del target_ds
    gc.collect()

# benchmark_one("1.0° Global", 180, 360)
# benchmark_one("0.25° Global", 720, 1440)
benchmark_one("0.1° Global", 1800, 3600)
