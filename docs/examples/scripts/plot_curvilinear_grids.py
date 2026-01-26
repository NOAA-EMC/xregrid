"""
Curvilinear Grid Regridding
===========================

This example demonstrates regridding from a curvilinear grid (where coordinates
are 2D arrays of latitude and longitude) to a standard rectilinear grid.
We use the 'rasm' tutorial dataset which features a curvilinear Arctic grid.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# Load rasm tutorial dataset (curvilinear Arctic grid)
ds = xr.tutorial.open_dataset("rasm").isel(time=0)
print(f"Source coordinates: xc (lon) and yc (lat) are {ds.xc.dims}")

# Define a standard rectilinear target grid (e.g., 1.0° global)
target_lat = np.arange(50, 91, 1.0)
target_lon = np.arange(0, 360, 1.0)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# XRegrid automatically detects the 2D coordinates in 'rasm'
regridder = Regridder(ds, target_grid, method="bilinear")

# Apply regridding
# We can regrid the whole dataset or just a DataArray
tair_regridded = regridder(ds.Tair)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot original (using its 2D coordinates)
ds.Tair.plot(ax=ax1, x="xc", y="yc")
ax1.set_title("Original Curvilinear Grid (rasm)")

# Plot regridded (rectilinear)
tair_regridded.plot(ax=ax2)
ax2.set_title("Regridded to Rectilinear 1.0°")

plt.tight_layout()
plt.show()
