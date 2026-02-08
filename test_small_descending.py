import xarray as xr
import numpy as np
import esmpy
from xregrid import Regridder

def test():
    # Source: lat descending [2, 1], lon ascending [10, 20]
    src_lat = [2, 1]
    src_lon = [10, 20]
    src_data = np.array([[20, 21], [10, 11]]) # data[0,0] is at lat=2, lon=10

    src_ds = xr.Dataset(
        {"air": (["lat", "lon"], src_data)},
        coords={"lat": src_lat, "lon": src_lon}
    )

    # Target: same as source but ascending lat [1, 2]
    tgt_lat = [1, 2]
    tgt_lon = [10, 20]
    tgt_grid_ds = xr.Dataset(
        coords={"lat": tgt_lat, "lon": tgt_lon}
    )

    regridder = Regridder(src_ds, tgt_grid_ds, method="bilinear")
    res = regridder(src_ds.air)

    print("Source data:")
    print(src_data)
    print("Regridded data:")
    print(res.values)

    # Expected: res[0, 0] (lat=1, lon=10) should be src[1, 0] = 10
    # Expected: res[1, 0] (lat=2, lon=10) should be src[0, 0] = 20

    if res.values[0, 0] == 10 and res.values[1, 0] == 20:
        print("SUCCESS: Small descending case works!")
    else:
        print("FAILURE: Small descending case failed!")

    print("\nWeights Matrix:")
    print(regridder._weights_matrix.toarray())
    # Indices in ESMF for src (2x2):
    # 0,0 (lon=10, lat=2) -> idx 1
    # 1,0 (lon=20, lat=2) -> idx 2
    # 0,1 (lon=10, lat=1) -> idx 3
    # 1,1 (lon=20, lat=1) -> idx 4

    # Indices in ESMF for tgt (2x2):
    # 0,0 (lon=10, lat=1) -> idx 1
    # 1,0 (lon=20, lat=1) -> idx 2
    # 0,1 (lon=10, lat=2) -> idx 3
    # 1,1 (lon=20, lat=2) -> idx 4

    # Mapping:
    # Tgt idx 1 (lat=1, lon=10) should come from Src idx 3 (lat=1, lon=10)
    # Tgt idx 2 (lat=1, lon=20) should come from Src idx 4 (lat=1, lon=20)
    # Tgt idx 3 (lat=2, lon=10) should come from Src idx 1 (lat=2, lon=10)
    # Tgt idx 4 (lat=2, lon=20) should come from Src idx 2 (lat=2, lon=20)

    # Expected Matrix:
    # [[0, 0, 1, 0],
    #  [0, 0, 0, 1],
    #  [1, 0, 0, 0],
    #  [0, 1, 0, 0]]

test()
