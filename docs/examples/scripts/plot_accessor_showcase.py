"""
Xarray Accessor Showcase
========================

This example demonstrates the high-level xarray accessor API provided by XRegrid.
The `.regrid.to()` method allows for concise and readable regridding workflows.

Key concepts demonstrated:
- Using the DataArray accessor: `da.regrid.to(target_grid)`
- Using the Dataset accessor: `ds.regrid.to(target_grid)`
- Passing Regridder parameters through the accessor
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load air_temperature tutorial dataset
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)

# Define a simple target grid (5.0° resolution)
target_lat = np.arange(15, 76, 5.0)
target_lon = np.arange(200, 331, 5.0)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# --- 1. DataArray Accessor ---
# You can call .regrid.to() directly on a DataArray
air_regridded = ds.air.regrid.to(target_grid, method="bilinear")

print(f"DataArray regridding successful. Shape: {air_regridded.shape}")

# --- 2. Dataset Accessor ---
# You can also call it on the entire Dataset
ds_regridded = ds.regrid.to(target_grid, method="bilinear")

print(f"Dataset regridding successful. Variables: {list(ds_regridded.data_vars)}")

# --- 3. Passing Parameters ---
# Any Regridder parameter can be passed through the accessor
# For example, using a different interpolation method
air_nearest = ds.air.regrid.to(target_grid, method="nearest_s2d")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ds.air.plot(ax=axes[0])
axes[0].set_title("Original (2.5°)")

air_regridded.plot(ax=axes[1])
axes[1].set_title("Bilinear (5.0°)")

air_nearest.plot(ax=axes[2])
axes[2].set_title("Nearest Neighbor (5.0°)")

plt.tight_layout()
plt.show()
