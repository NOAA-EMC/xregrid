import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

def test():
    # Source grid
    lat = np.linspace(-90, 90, 19)
    lon = np.linspace(0, 360, 37)
    lon_m, lat_m = np.meshgrid(lon, lat)
    data = np.sin(np.deg2rad(lat_m)) * np.cos(np.deg2rad(lon_m))

    src_ds = xr.Dataset(
        {"wave": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon}
    )

    # Target grid (finer)
    t_lat = np.linspace(-90, 90, 37)
    t_lon = np.linspace(0, 360, 73)
    tgt_ds = xr.Dataset(
        coords={"lat": t_lat, "lon": t_lon}
    )

    regridder = Regridder(src_ds, tgt_ds, method="bilinear", periodic=True)
    res = regridder(src_ds.wave)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    src_ds.wave.plot(ax=ax1)
    ax1.set_title("Original")
    res.plot(ax=ax2)
    ax2.set_title("Regridded")
    plt.savefig("test_wave.png")

    # Check max difference (at source points)
    res_at_src = res.sel(lat=lat, lon=lon, method='nearest')
    diff = np.abs(res_at_src.values - data)
    print(f"Max difference at source points: {np.max(diff)}")

test()
