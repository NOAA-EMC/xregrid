import xarray as xr
import numpy as np
import esmpy
from xregrid import Regridder

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
    # Source
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

    # Target
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

    # Regrid
    regrid = esmpy.Regrid(src_field, dst_field, regrid_method=esmpy.RegridMethod.BILINEAR, unmapped_action=esmpy.UnmappedAction.IGNORE)
    regrid(src_field, dst_field)

    air_esmpy_vals = dst_field.data.T

    # Compare
    diff = np.abs(air_xregrid.values - air_esmpy_vals)
    print(f"Max difference between XRegrid and Raw ESMPy: {np.max(diff)}")

    if np.max(diff) > 1e-5:
        print("DIFFERENCE DETECTED!")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        air_xregrid.plot(ax=ax1)
        ax1.set_title("XRegrid")
        ax2.imshow(air_esmpy_vals, origin='lower')
        ax2.set_title("Raw ESMPy (imshow)")
        ax3.imshow(diff, origin='lower')
        ax3.set_title("Difference")
        plt.savefig("comparison_debug.png")
    else:
        print("XRegrid matches Raw ESMPy!")

test()
