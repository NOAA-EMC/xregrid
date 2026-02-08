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
matrix = regridder._weights_matrix

# Pick a target point: lat=40, lon=250
# target_lat[25] = 40
# target_lon[50] = 250
dst_lat_idx = 25
dst_lon_idx = 50
n_lon_dst = 131
dst_idx = dst_lat_idx * n_lon_dst + dst_lon_idx

print(f"Checking target point: lat={target_lat[dst_lat_idx]}, lon={target_lon[dst_lon_idx]}")
print(f"Destination index: {dst_idx}")

# Get weights for this row
row = matrix.getrow(dst_idx)
print(f"Number of non-zero weights: {row.nnz}")

for i in range(row.nnz):
    src_idx = row.indices[i]
    weight = row.data[i]

    # Map src_idx back to (lat, lon)
    n_lon_src = 53
    src_lat_idx = src_idx // n_lon_src
    src_lon_idx = src_idx % n_lon_src

    src_lat_val = ds.lat.values[src_lat_idx]
    src_lon_val = ds.lon.values[src_lon_idx]

    print(f"  Source index {src_idx}: lat={src_lat_val}, lon={src_lon_val}, weight={weight}")
