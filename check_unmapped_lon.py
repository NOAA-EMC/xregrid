import xarray as xr
import numpy as np
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

regridder = Regridder(ds, target_grid_ds, method="bilinear")
air_regridded = regridder(ds.air)

lat15 = air_regridded.sel(lat=15.0)
unmapped = np.where(lat15.values == 0)[0]
mapped = np.where(lat15.values != 0)[0]

print(f"At lat=15.0:")
print(f"  Number of unmapped: {len(unmapped)}")
print(f"  Number of mapped: {len(mapped)}")
print(f"  Mapped longitudes: {lat15.lon.values[mapped]}")
