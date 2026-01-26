"""
Performance Optimization: Weight Reuse
======================================

One of XRegrid's biggest performance advantages is the ability to generate
regridding weights once and reuse them for multiple datasets or time steps.
This example demonstrates how to save and load weights.
"""

import xarray as xr
import numpy as np
import time
import os
from xregrid import Regridder

# Load a larger tutorial dataset
ds = xr.tutorial.open_dataset("air_temperature")

# Target grid
target_lat = np.arange(ds.lat.min().values, ds.lat.max().values, 1.0)
target_lon = np.arange(ds.lon.min().values, ds.lon.max().values, 1.0)
target_grid = xr.Dataset({"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)})

weights_file = "weights.nc"
print(f"Dataset size: {ds.air.nbytes / 1e6:.2f} MB")

# 1. First time: Generate and save weights
start = time.time()
regridder = Regridder(
    ds, target_grid, method="bilinear", filename=weights_file, reuse_weights=True
)
_ = regridder(ds.air)
first_time = time.time() - start
print(f"First run (with weight generation): {first_time:.2f}s")

# 2. Second time: Load weights from disk
start = time.time()
regridder_cached = Regridder(
    ds, target_grid, method="bilinear", filename=weights_file, reuse_weights=True
)
_ = regridder_cached(ds.air)
second_time = time.time() - start
print(f"Second run (reusing weights): {second_time:.2f}s")

print(f"\nSpeedup: {first_time / second_time:.1f}x")

# Clean up
if os.path.exists(weights_file):
    os.remove(weights_file)
