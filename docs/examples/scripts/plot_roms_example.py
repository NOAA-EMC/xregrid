"""
ROMS Curvilinear Grid Regridding
================================

This example demonstrates regridding from a ROMS (Regional Ocean Modeling System)
curvilinear grid to a standard rectilinear grid. ROMS grids use 2D latitude
and longitude arrays.

Key concepts demonstrated:
- Handling ocean model curvilinear grids
- Automatic detection of coordinates via cf-xarray
- Regridding entire datasets with multiple variables
- Preserving temporal and vertical dimensions
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# Load ROMS_example tutorial dataset
ds = xr.tutorial.open_dataset("ROMS_example")

print(f"Source Dataset: {list(ds.data_vars)}")
print(f"Source coordinates: lon_rho and lat_rho are {ds.lon_rho.dims}")
print(f"Dataset shape: {ds.salt.shape} (ocean_time, s_rho, eta_rho, xi_rho)")

# Define a rectilinear target grid for the Gulf of Mexico region
target_lat = np.arange(26, 31, 0.1)
target_lon = np.arange(-98, -88, 0.1)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# XRegrid uses cf-xarray to find 'lat_rho' and 'lon_rho' via their standard_name attributes
regridder = Regridder(ds, target_grid, method="bilinear")

# Regrid the entire dataset
# This will regrid 'salt' (4D) and 'zeta' (3D) while preserving non-spatial dimensions
ds_regridded = regridder(ds)

# Visualization of surface salinity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot original curvilinear data (first time step, surface level)
ds.salt.isel(ocean_time=0, s_rho=-1).plot(
    ax=ax1, x="lon_rho", y="lat_rho", cmap="viridis"
)
ax1.set_title("Original ROMS Salt (Surface)")

# Plot regridded rectilinear data
ds_regridded.salt.isel(ocean_time=0, s_rho=-1).plot(ax=ax2, cmap="viridis")
ax2.set_title("Regridded ROMS Salt (Surface)")

plt.tight_layout()
plt.show()

print("\nRegridding Summary:")
print(f"Output Dataset variables: {list(ds_regridded.data_vars)}")
print(f"Output salt shape: {ds_regridded.salt.shape} (ocean_time, s_rho, lat, lon)")
print(f"Output zeta shape: {ds_regridded.zeta.shape} (ocean_time, lat, lon)")
