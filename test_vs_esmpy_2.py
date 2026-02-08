import xarray as xr
import numpy as np
import esmpy
from xregrid import Regridder
import matplotlib.pyplot as plt

def test():
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
    target_lat = np.arange(15, 76, 1.0)
    target_lon = np.arange(200, 331, 1.0)
    target_grid_ds = xr.Dataset(
        {
            "lat": (["lat"], target_lat, {"units": "degrees_north"}),
            "lon": (["lon"], target_lon, {"units": "degrees_east"}),
        }
    )

    # 1. XRegrid
    regridder = Regridder(ds, target_grid_ds, method="bilinear")
    air_xregrid = regridder(ds.air)

    # 2. Raw ESMPy
    src_grid = esmpy.Grid(
        np.array([ds.lon.size, ds.lat.size]),
        staggerloc=[esmpy.StaggerLoc.CENTER],
        coord_sys=esmpy.CoordSys.SPH_DEG
    )
    src_lon_ptr = src_grid.get_coords(0)
    src_lat_ptr = src_grid.get_coords(1)
    lon_mesh_src, lat_mesh_src = np.meshgrid(ds.lon.values, ds.lat.values)
    src_lon_ptr[...] = lon_mesh_src.T
    src_lat_ptr[...] = lat_mesh_src.T

    src_field = esmpy.Field(src_grid, name="air")
    src_field.data[...] = ds.air.values.T

    dst_grid = esmpy.Grid(
        np.array([len(target_lon), len(target_lat)]),
        staggerloc=[esmpy.StaggerLoc.CENTER],
        coord_sys=esmpy.CoordSys.SPH_DEG
    )
    dst_lon_ptr = dst_grid.get_coords(0)
    dst_lat_ptr = dst_grid.get_coords(1)
    lon_mesh_dst, lat_mesh_dst = np.meshgrid(target_lon, target_lat)
    dst_lon_ptr[...] = lon_mesh_dst.T
    dst_lat_ptr[...] = lat_mesh_dst.T

    dst_field = esmpy.Field(dst_grid, name="air_regridded")

    regrid = esmpy.Regrid(src_field, dst_field, regrid_method=esmpy.RegridMethod.BILINEAR, unmapped_action=esmpy.UnmappedAction.IGNORE)
    regrid(src_field, dst_field)

    air_esmpy_vals = dst_field.data.T

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ds.air.plot(ax=ax1)
    ax1.set_title("Original")

    air_xregrid.plot(ax=ax2)
    ax2.set_title("XRegrid")

    # Plot ESMPy result as a DataArray to use xarray's plotting
    esmpy_da = xr.DataArray(air_esmpy_vals, coords={"lat": target_lat, "lon": target_lon}, dims=("lat", "lon"))
    esmpy_da.plot(ax=ax3)
    ax3.set_title("Raw ESMPy")

    plt.savefig("comparison_full.png")
    print("Plot saved to comparison_full.png")

    print(f"Original range: {ds.air.min().values}, {ds.air.max().values}")
    print(f"XRegrid range: {air_xregrid.min().values}, {air_xregrid.max().values}")
    print(f"ESMPy range: {esmpy_da.min().values}, {esmpy_da.max().values}")

test()
