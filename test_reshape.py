import numpy as np

n_lat = 25
n_lon = 53
data = np.random.rand(n_lat, n_lon)

flat_data = data.reshape(1, -1)

lat_idx = 14
lon_idx = 20
k = lat_idx * n_lon + lon_idx

print(f"data[{lat_idx}, {lon_idx}] = {data[lat_idx, lon_idx]}")
print(f"flat_data[0, {k}] = {flat_data[0, k]}")

if data[lat_idx, lon_idx] == flat_data[0, k]:
    print("MATCH!")
else:
    print("MISMATCH!")
