import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder
import esmpy

def test():
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
    target_lat = np.arange(15, 76, 1.0)
    target_lon = np.arange(200, 331, 1.0)

    ds_with_bounds = ds.cf.add_bounds(['latitude', 'longitude'])
    target_grid_ds = xr.Dataset(
        coords={
            "lat": (["lat"], target_lat, {"units": "degrees_north"}),
            "lon": (["lon"], target_lon, {"units": "degrees_east"}),
        }
    ).cf.add_bounds(['latitude', 'longitude'])

    regridder = Regridder(ds_with_bounds, target_grid_ds, method="conservative")
    res = regridder(ds.air)

    zeros = np.sum(res.values == 0)
    print(f"Zeros with conservative and SPH_DEG: {zeros}")
    print(f"Range: {res.min().values}, {res.max().values}")

test()
