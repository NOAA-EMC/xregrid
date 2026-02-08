import xarray as xr
ds = xr.tutorial.open_dataset("air_temperature")
print("Latitudes:")
print(ds.lat.values)
print("\nLongitudes:")
print(ds.lon.values)
