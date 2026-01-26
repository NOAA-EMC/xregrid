"""
Basic Rectilinear Grid Regridding
=================================

This example demonstrates the most common use case: regridding between
rectilinear latitude-longitude grids. We use the ERA-Interim tutorial
dataset and regrid it to a coarser resolution.

Key concepts demonstrated:
- Loading global tutorial datasets
- Creating a target grid
- Using the Regridder with bilinear method
- Handling global periodicity
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# Load tutorial dataset (global, 0.75° resolution)
ds = xr.tutorial.open_dataset("eraint_uvz").isel(month=0, level=0)
print(f"Source grid shape: {ds.u.shape}")

# Create a coarser target grid (2.0° resolution)
target_lat = np.arange(-90, 91, 2.0)
target_lon = np.arange(-180, 180, 2.0)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# Create the regridder
# periodic=True is essential for global grids to handle the dateline
regridder = Regridder(ds, target_grid, method="bilinear", periodic=True)

# Apply regridding
u_regridded = regridder(ds.u)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ds.u.plot(ax=ax1, cmap="RdBu_r")
ax1.set_title(f"Original ERA-Interim (0.75°)\nShape: {ds.u.shape}")

u_regridded.plot(ax=ax2, cmap="RdBu_r")
ax2.set_title(f"Regridded to 2.0°\nShape: {u_regridded.shape}")

plt.tight_layout()
plt.show()

# Summary of the operation
print("\nRegridding Summary:")
print(f"Method: {regridder.method}")
print(f"Periodic: {regridder.periodic}")
print(f"Input dimensions: {ds.u.dims}")
print(f"Output dimensions: {u_regridded.dims}")
