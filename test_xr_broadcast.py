import xarray as xr
import numpy as np

lon = xr.DataArray([10, 20, 30], coords={'lon': [10, 20, 30]}, dims='lon', name='lon')
lat = xr.DataArray([1, 2], coords={'lat': [1, 2]}, dims='lat', name='lat')

print("Broadcasting (lon, lat):")
lon_m, lat_m = xr.broadcast(lon, lat)
print(f"lon_m dims: {lon_m.dims}")
print(f"lat_m dims: {lat_m.dims}")

print("\nBroadcasting (lat, lon):")
lat_m2, lon_m2 = xr.broadcast(lat, lon)
print(f"lon_m2 dims: {lon_m2.dims}")
print(f"lat_m2 dims: {lat_m2.dims}")

ds = xr.Dataset({'air': (('lat', 'lon'), np.random.rand(2, 3))},
                coords={'lat': [1, 2], 'lon': [10, 20, 30]})
print("\nDataset dimensions order:")
print(ds.air.dims)
lon_ds = ds.lon
lat_ds = ds.lat
lon_m3, lat_m3 = xr.broadcast(lon_ds, lat_ds)
print(f"lon_m3 dims from dataset: {lon_m3.dims}")
