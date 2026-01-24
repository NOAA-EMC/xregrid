import sys
import time
import numpy as np
from unittest.mock import MagicMock

# Mock esmpy for both xregrid and xesmf
mock_esmpy = MagicMock()
sys.modules["esmpy"] = mock_esmpy

import xregrid
import xesmf
import sparse
import scipy.sparse as sp
import xarray as xr

# xesmf.smm.apply_weights doesn't actually need esmpy to run, it just needs it to import
from xesmf.smm import apply_weights as xesmf_apply_weights

def benchmark_resolution(name, nlat_in, nlon_in, nlat_out, nlon_out, trials=10):
    n_in = nlat_in * nlon_in
    n_out = nlat_out * nlon_out

    # Create random input data (DataArray for realistic overhead)
    data_in_np = np.random.rand(nlat_in, nlon_in)
    da_in = xr.DataArray(data_in_np, dims=("lat", "lon"))

    # Create weight matrix for bilinear (approx 4 non-zeros per row)
    nnz = n_out * 4
    rows = np.repeat(np.arange(n_out), 4)
    cols = np.random.randint(0, n_in, nnz)
    weights_data = np.random.rand(nnz)

    # --- XRegrid Setup ---
    # We'll use the internal _apply_weights of ESMPyRegridder but bypass weight generation
    regridder = xregrid.ESMPyRegridder.__new__(xregrid.ESMPyRegridder)
    regridder._weights_matrix = sp.coo_matrix((weights_data, (rows, cols)), shape=(n_out, n_in))
    regridder._shape_source = (nlat_in, nlon_in)
    regridder._shape_target = (nlat_out, nlon_out)
    regridder._dims_source = ("lat", "lon")
    regridder._dims_target = ("lat", "lon")
    regridder.skipna = False
    regridder.periodic = False
    regridder.method = "bilinear"
    regridder.target_grid_ds = xr.Dataset(coords={
        "lat": (["lat"], np.linspace(-90, 90, nlat_out)),
        "lon": (["lon"], np.linspace(0, 360, nlon_out))
    })

    # --- xESMF Setup ---
    # xESMF weights are (n_out, n_in) in smm.apply_weights
    xesmf_weights = sparse.COO(coords=np.vstack([rows, cols]), data=weights_data, shape=(n_out, n_in))
    # But wait, as I saw in Regridder.regrid_array, it reshapes it to (shape_out + shape_in)
    xesmf_weights_reshaped = xesmf_weights.reshape((nlat_out, nlon_out, nlat_in, nlon_in))

    print(f"\n--- Benchmarking {name} ---")
    print(f"Source: {nlat_in}x{nlon_in} ({n_in} pts), Target: {nlat_out}x{nlon_out} ({n_out} pts)")

    # Warmup
    _ = regridder(da_in)
    _ = xesmf_apply_weights(xesmf_weights_reshaped, data_in_np, (nlat_in, nlon_in), (nlat_out, nlon_out))

    # XRegrid Benchmark
    times_xregrid = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = regridder(da_in)
        times_xregrid.append(time.perf_counter() - start)

    avg_xregrid = np.mean(times_xregrid)
    print(f"XRegrid avg: {avg_xregrid:.6f} s")

    # xESMF Benchmark
    times_xesmf = []
    for _ in range(trials):
        start = time.perf_counter()
        _ = xesmf_apply_weights(xesmf_weights_reshaped, data_in_np, (nlat_in, nlon_in), (nlat_out, nlon_out))
        times_xesmf.append(time.perf_counter() - start)

    avg_xesmf = np.mean(times_xesmf)
    print(f"xESMF avg: {avg_xesmf:.6f} s")
    print(f"Speedup: {avg_xesmf / avg_xregrid:.1f}x")

# 1.0° Global
benchmark_resolution("1.0° Global", 180, 360, 180, 360)

# 0.25° Global
benchmark_resolution("0.25° Global", 720, 1440, 720, 1440)

# 0.1° Global
benchmark_resolution("0.1° Global", 1800, 3600, 1800, 3600, trials=3)
