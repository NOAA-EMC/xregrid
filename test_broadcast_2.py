
import xarray as xr
import numpy as np

def test_broadcast_values():
    lon = xr.DataArray([10, 20, 30], dims='lon', name='lon')
    lat = xr.DataArray([1, 2], dims='lat', name='lat')

    lon_mesh, lat_mesh = xr.broadcast(lon, lat)
    # lon_mesh is ('lon', 'lat') shape (3, 2)
    # [[10, 10],
    #  [20, 20],
    #  [30, 30]]

    lon_mesh = lon_mesh.transpose('lat', 'lon')
    # shape (2, 3)
    # [[10, 20, 30],
    #  [10, 20, 30]]
    print("lon_mesh values:\n", lon_mesh.values)

    lat_mesh = lat_mesh.transpose('lat', 'lon')
    # [[1, 1, 1],
    #  [2, 2, 2]]
    print("lat_mesh values:\n", lat_mesh.values)

if __name__ == "__main__":
    test_broadcast_values()
