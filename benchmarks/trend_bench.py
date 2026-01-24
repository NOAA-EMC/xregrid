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

    # Trials
    t_xesmf_list = []
    t_xregrid_list = []
    for _ in range(5):
        start = time.perf_counter()
        _ = regridder_xesmf(data_in)
        t_xesmf_list.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = regridder_xregrid(data_in)
        t_xregrid_list.append(time.perf_counter() - start)

    t_xesmf = np.mean(t_xesmf_list)
    t_xregrid = np.mean(t_xregrid_list)

    print(f"xESMF: {t_xesmf:.6f}s, XRegrid: {t_xregrid:.6f}s")
    print(f"Speedup: {t_xesmf / t_xregrid:.1f}x")

    del regridder_xesmf
    del regridder_xregrid
    del source_ds
    del target_ds
    gc.collect()

benchmark_one("1.0° Global", 180, 360)
benchmark_one("0.5° Global", 360, 720)
benchmark_one("0.25° Global", 720, 1440)
