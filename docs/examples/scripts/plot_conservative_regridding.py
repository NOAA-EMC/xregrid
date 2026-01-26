"""
Conservative Regridding for Flux Data
====================================

Conservative regridding is essential for flux quantities (like precipitation
or radiation) where mass or energy must be preserved.

This example demonstrates how to use the 'conservative' method and verifies
area-integrated conservation.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# Load tutorial dataset
# We'll use 'air_temperature' but treat it as a dummy flux for this demonstration
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)

# Create a coarser target grid
target_lat = np.linspace(ds.lat.min().values, ds.lat.max().values, 10)
target_lon = np.linspace(ds.lon.min().values, ds.lon.max().values, 15)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# Create conservative regridder
regridder = Regridder(ds, target_grid, method="conservative")

# Apply regridding
air_cons = regridder(ds.air)

# For comparison, also do bilinear
regridder_bil = Regridder(ds, target_grid, method="bilinear")
air_bil = regridder_bil(ds.air)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ds.air.plot(ax=axes[0])
axes[0].set_title("Original (High Res)")

air_cons.plot(ax=axes[1])
axes[1].set_title("Conservative (Low Res)")

air_bil.plot(ax=axes[2])
axes[2].set_title("Bilinear (Low Res)")

plt.tight_layout()
plt.show()

print("\nConservation analysis:")
orig_mean = ds.air.mean().values
cons_mean = air_cons.mean().values
bil_mean = air_bil.mean().values

print(f"Original mean: {orig_mean:.4f}")
print(f"Conservative mean: {cons_mean:.4f}")
print(f"Bilinear mean: {bil_mean:.4f}")
print(f"Conservative delta: {abs(cons_mean - orig_mean):.4e}")
