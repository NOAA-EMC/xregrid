"""
Unstructured Grid Regridding
============================

This example demonstrates regridding from an unstructured grid (e.g., MPAS or ICON)
to a standard rectilinear grid. Unstructured grids are characterized by having
a single dimension for all spatial locations, rather than separate latitude
and longitude dimensions.

Key concepts:
- Handling 1D spatial coordinates
- Point-to-grid regridding via LocStream
- Nearest neighbor interpolation methods
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# 1. Create a synthetic unstructured grid
# For example, 1000 cells at random locations
np.random.seed(42)
n_cells = 1000
lats = np.random.uniform(-90, 90, n_cells)
lons = np.random.uniform(0, 360, n_cells)

# Create some synthetic data (e.g., a simple wave pattern)
data_vals = np.sin(np.radians(lats)) * np.cos(np.radians(lons))

ds_unstructured = xr.Dataset(
    {"data": (["nCells"], data_vals)},
    coords={
        "lat": (["nCells"], lats, {"units": "degrees_north"}),
        "lon": (["nCells"], lons, {"units": "degrees_east"}),
    },
)

# 2. Define a standard rectilinear target grid (e.g., 4° global)
target_lat = np.arange(-90, 91, 4.0)
target_lon = np.arange(0, 361, 4.0)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# 3. Create the regridder
# For unstructured source data without connectivity, we use nearest-neighbor
regridder = Regridder(ds_unstructured, target_grid, method="nearest_s2d")

# 4. Apply regridding
ds_regridded = regridder(ds_unstructured)

# 5. Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot unstructured points
sc = ax1.scatter(
    ds_unstructured.lon,
    ds_unstructured.lat,
    c=ds_unstructured.data,
    s=30,
    cmap="viridis",
    edgecolor="none",
    alpha=0.7,
)
ax1.set_title(f"Unstructured Source Points ({n_cells} cells)")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
plt.colorbar(sc, ax=ax1, label="Data Value")

# Plot regridded result
ds_regridded.data.plot(ax=ax2, cmap="viridis")
ax2.set_title("Regridded to Rectilinear Grid (4°)")

plt.tight_layout()
plt.show()

print("\nRegridding Summary:")
print(f"Source: Unstructured ({n_cells} cells)")
print(f"Target: Rectilinear ({len(target_lat)}x{len(target_lon)})")
print(f"Method: {regridder.method}")
