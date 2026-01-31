"""
Large Dataset Visualization Example (Aero Protocol)
===================================================

This example demonstrates how to visualize extremely large regridded datasets
using the two-track approach:
1. Track A: Static publication-quality plots using Matplotlib + Cartopy.
2. Track B: Interactive exploration using HvPlot + GeoViews with rasterization.
"""

import xarray as xr
import numpy as np
from xregrid import Regridder, plot
from xregrid.utils import create_global_grid

# 1. Create a "large-ish" synthetic dataset (simulating ~1km or high-res)
# For the sake of this example, we'll use a 0.25 degree grid.
ds_src = create_global_grid(0.25, 0.25)
ds_tgt = create_global_grid(0.1, 0.1)

# Create some synthetic data (e.g., a wave pattern)
lat = ds_src.lat.values
lon = ds_src.lon.values
lons, lats = np.meshgrid(lon, lat)
data = np.sin(np.deg2rad(lons)) * np.cos(np.deg2rad(lats))

da = xr.DataArray(
    data,
    coords={"lat": ds_src.lat, "lon": ds_src.lon},
    dims=("lat", "lon"),
    name="wave_pattern",
    attrs={"units": "m", "long_name": "Synthetic Wave Pattern"}
)

# 2. Regrid the data
regridder = Regridder(ds_src, ds_tgt, method='bilinear')
da_regridded = regridder(da)

# --- TRACK A: Publication (Static) ---
# plot() with mode='static' (default) uses matplotlib + cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(12, 6))
# plot_static is called internally by xregrid.plot
im = plot(
    da_regridded,
    mode='static',
    projection=ccrs.Robinson(),
    cmap='viridis',
    title="Global 0.1° Wave Pattern (Track A: Static)"
)
plt.savefig("large_dataset_static.png", bbox_inches='tight', dpi=300)
print("Saved static plot to large_dataset_static.png")

# --- TRACK B: Exploration (Interactive) ---
# plot() with mode='interactive' uses hvplot with rasterize=True for large grids
try:
    interactive_plot = plot(
        da_regridded,
        mode='interactive',
        width=800,
        height=400,
        title="Interactive Global Grid (Track B: Interactive)"
    )
    # In a notebook, this would display the interactive widget.
    # For this script, we just acknowledge its creation.
    print("Interactive plot object created successfully (Track B).")
except ImportError:
    print("Skipping interactive plot (hvplot not installed).")

# --- COMPARISON ---
from xregrid.viz import plot_comparison
# plot_comparison allows side-by-side view with difference
fig_comp = plot_comparison(
    da,
    da_regridded,
    regridder=regridder,
    title="Regridding Comparison: 0.25° to 0.1°"
)
plt.savefig("regrid_comparison.png", bbox_inches='tight')
print("Saved comparison plot to regrid_comparison.png")
