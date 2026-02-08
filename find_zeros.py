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

zeros = np.where(air_regridded.values == 0)
print(f"Number of zeros: {len(zeros[0])}")

if len(zeros[0]) > 0:
    for i in range(min(10, len(zeros[0]))):
        idx_lat = zeros[0][i]
        idx_lon = zeros[1][i]
        print(f"Zero at: lat={air_regridded.lat.values[idx_lat]}, lon={air_regridded.lon.values[idx_lon]}")
