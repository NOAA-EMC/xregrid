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

# Value at lat=40, lon=250
val_regridded = air_regridded.sel(lat=40, lon=250).values
val_original = ds.air.sel(lat=40, lon=250).values

print(f"At lat=40, lon=250:")
print(f"  Original value: {val_original}")
print(f"  Regridded value: {val_regridded}")

# Value at lat=75, lon=200
val_regridded_top = air_regridded.sel(lat=75, lon=200).values
val_original_top = ds.air.sel(lat=75, lon=200).values

print(f"\nAt lat=75, lon=200:")
print(f"  Original value: {val_original_top}")
print(f"  Regridded value: {val_regridded_top}")

# Check if the result is flipped
val_original_bottom = ds.air.sel(lat=15, lon=200).values
print(f"\nOriginal value at lat=15, lon=200: {val_original_bottom}")
