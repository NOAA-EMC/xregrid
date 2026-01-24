import time
import numpy as np
import xarray as xr
from xregrid import ESMPyRegridder
import esmpy

def test_unstructured_nearest():
    print("Testing unstructured grid support (nearest)...")
    ncells_src = 1000
    ncells_tgt = 2000

    src_ds = xr.Dataset({
        "lat": (["nCells"], np.linspace(-90, 90, ncells_src)),
        "lon": (["nCells"], np.linspace(0, 360, ncells_src))
    })

    tgt_ds = xr.Dataset({
        "lat": (["nCells"], np.linspace(-90, 90, ncells_tgt)),
        "lon": (["nCells"], np.linspace(0, 360, ncells_tgt))
    })

    data = xr.DataArray(np.random.rand(ncells_src), dims="nCells", coords={"lat": src_ds.lat, "lon": src_ds.lon})

    try:
        # Nearest neighbor should work with LocStream
        regridder = ESMPyRegridder(src_ds, tgt_ds, method="nearest_s2d")
        out = regridder(data)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_unstructured_nearest()
