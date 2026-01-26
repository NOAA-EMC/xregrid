"""
Weather Data: Station to Grid
============================

This example leverages the toy weather dataset concept from the xarray
documentation and shows how to regrid station-like data (points) to a
regular 2D grid.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xregrid import Regridder

# 1. Create toy weather data (similar to xarray docs)
np.random.seed(123)
times = pd.date_range("2000-01-01", periods=10, name="time")
locations = ["IA", "IN", "IL"]

# Assign lat/lon to these locations
loc_lats = [42.0, 40.0, 40.0]
loc_lons = [267.0, 274.0, 271.0]

ds = xr.Dataset(
    {
        "tmin": (("time", "location"), np.random.randn(10, 3) + 10),
        "tmax": (("time", "location"), np.random.randn(10, 3) + 20),
    },
    {
        "time": times,
        "location": locations,
        "lat": (("location",), loc_lats, {"units": "degrees_north"}),
        "lon": (("location",), loc_lons, {"units": "degrees_east"}),
    },
)

print("Source Dataset (Station-like):")
print(ds)

# 2. Define a target grid for the region
target_lat = np.linspace(38, 44, 50)
target_lon = np.linspace(265, 276, 50)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# 3. Regrid from points to grid
# For points, we use nearest-neighbor methods
regridder = Regridder(ds, target_grid, method="nearest_s2d")

# Apply regridding to the whole dataset
ds_regridded = regridder(ds)

# 4. Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot stations as points
sc = ax1.scatter(
    ds.lon, ds.lat, c=ds.tmax.isel(time=0), s=200, cmap="viridis", edgecolor="k"
)
plt.colorbar(sc, ax=ax1, label="Temperature")
ax1.set_title("Station Data (tmax, day 0)")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.grid(True, linestyle="--", alpha=0.5)

# Plot regridded grid
ds_regridded.tmax.isel(time=0).plot(ax=ax2, cmap="viridis")
ax2.set_title("Regridded to 2D Grid (Nearest Neighbor)")

plt.tight_layout()
plt.show()
