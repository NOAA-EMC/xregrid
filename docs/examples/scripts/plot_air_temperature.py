"""
Air Temperature Dataset Regridding
==================================

This example demonstrates regridding the 'air_temperature' tutorial dataset
from its native 2.5° resolution to a finer 1.0° resolution.

Key concepts demonstrated:
- Using xr.tutorial datasets
- Upsampling to a higher resolution
- Automatic CF-compliant coordinate detection
- Regridding 3D datasets (time, lat, lon)
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# Load air_temperature tutorial dataset (North America, 2.5° resolution)
ds = xr.tutorial.open_dataset("air_temperature")
print(f"Source Dataset variables: {list(ds.data_vars)}")
print(f"Source grid: {ds.lat.size}x{ds.lon.size} (2.5° resolution)")
print(f"Dataset shape: {ds.air.shape} (time, lat, lon)")

# Define a finer target grid (1.0° resolution) for the same region
# Note: air_temperature lon is 200 to 330
target_lat = np.arange(15, 76, 1.0)
target_lon = np.arange(200, 331, 1.0)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# Create the regridder
regridder = Regridder(ds, target_grid, method="bilinear")

# Apply regridding to the whole dataset (all time steps)
ds_regridded = regridder(ds)

# Visualization of the first time step
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ds.air.isel(time=0).plot(ax=ax1, cmap="RdYlBu_r")
ax1.set_title(f"Original Air Temp (2.5°)\nTime: {ds.time.values[0]}")

ds_regridded.air.isel(time=0).plot(ax=ax2, cmap="RdYlBu_r")
ax2.set_title(f"Regridded to 1.0°\nTime: {ds_regridded.time.values[0]}")

plt.tight_layout()
plt.show()

# Verify the output
print("\nRegridding Summary:")
print(f"Output air shape: {ds_regridded.air.shape} (time, lat, lon)")
print(f"Original title: {ds.attrs.get('title')}")
print(f"Regridded title: {ds_regridded.attrs.get('title')}")
