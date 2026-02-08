import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
target_lat = np.arange(15, 76, 1.0)
target_lon = np.arange(200, 331, 1.0)
target_grid_ds = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# Use extrapolation
regridder = Regridder(ds, target_grid_ds, method="bilinear", extrap_method="nearest_s2d")
air_regridded = regridder(ds.air)

zeros = np.sum(air_regridded.values == 0)
print(f"Zeros with extrapolation: {zeros}")
print(f"Range: {air_regridded.min().values}, {air_regridded.max().values}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ds.air.plot(ax=ax1, cmap="magma")
air_regridded.plot(ax=ax2, cmap="magma")
plt.savefig("test_extrap.png")
