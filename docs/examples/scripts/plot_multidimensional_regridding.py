"""
Multidimensional (4D) Regridding
================================

This example demonstrates regridding a 4D rectilinear dataset containing
multiple variables (u, v, z) and multiple non-spatial dimensions (month, level).

Key concepts demonstrated:
- Regridding 4D Datasets
- Handling multiple variables simultaneously
- Preservation of non-spatial dimensions (month, level)
- Global periodicity for 4D atmospheric data
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# Load multidimensional tutorial dataset (ERA-Interim)
# Dimensions: (month: 2, level: 3, latitude: 241, longitude: 480)
ds = xr.tutorial.open_dataset("eraint_uvz")

print("Source Dataset information:")
print(ds)
print(f"\nVariables to regrid: {list(ds.data_vars)}")
print(f"Data shape (e.g., 'u'): {ds.u.shape} (month, level, lat, lon)")

# Define a coarser target grid (5.0° resolution)
target_lat = np.arange(-90, 91, 5.0)
target_lon = np.arange(-180, 180, 5.0)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# Create the regridder
# periodic=True is essential for global atmospheric datasets
regridder = Regridder(ds, target_grid, method="bilinear", periodic=True)

# Apply regridding to the entire 4D Dataset
ds_regridded = regridder(ds)

# Visualization of 'u' wind at a specific level and month
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Original data
ds.u.isel(month=0, level=0).plot(ax=ax1, cmap="RdBu_r")
ax1.set_title(f"Original U-Wind (0.75°)\nLevel: {ds.level.values[0]} hPa")

# Regridded data
ds_regridded.u.isel(month=0, level=0).plot(ax=ax2, cmap="RdBu_r")
ax2.set_title(f"Regridded U-Wind (5.0°)\nLevel: {ds_regridded.level.values[0]} hPa")

plt.tight_layout()
plt.show()

# Verify the result
print("\nRegridding Summary:")
print(f"Output Dataset variables: {list(ds_regridded.data_vars)}")
print(f"Output 'u' shape: {ds_regridded.u.shape} (month, level, lat, lon)")
print(f"Output 'z' shape: {ds_regridded.z.shape} (month, level, lat, lon)")

# Check if non-spatial coordinates are preserved
xr.testing.assert_identical(ds.month, ds_regridded.month)
xr.testing.assert_identical(ds.level, ds_regridded.level)
print("\nNon-spatial dimensions (month, level) were correctly preserved.")
