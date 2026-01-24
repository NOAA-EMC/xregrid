import time
import numpy as np
import xarray as xr
import xesmf as xe
import gc

def create_sample_dataset(nlat, nlon):
    lat = np.linspace(-89.9, 89.9, nlat)
    lon = np.linspace(0, 359.5, nlon)
    data = np.random.rand(nlat, nlon).astype(np.float32)
    ds = xr.Dataset(
        {"temperature": (["lat", "lon"], data)},
        coords={"lat": (["lat"], lat), "lon": (["lon"], lon)},
    )
    return ds

print("Starting xESMF 0.1 deg...")
n = 1800
m = 3600
ds = create_sample_dataset(n, m)
regridder = xe.Regridder(ds, ds, method="bilinear", periodic=True)
print("Weight generation done for xESMF")
_ = regridder(ds["temperature"])
print("Application done for xESMF")
