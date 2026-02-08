
import sys
import numpy as np
import xarray as xr

class MockEsmpy:
    class CoordSys: SPH_DEG = 1
    class StaggerLoc: CENTER = 0; CORNER = 1
    class GridItem: MASK = 1
    class RegridMethod: BILINEAR = 0; CONSERVE = 1; NEAREST_STOD = 2; NEAREST_DTOS = 3; PATCH = 4
    class UnmappedAction: IGNORE = 1
    class ExtrapMethod: NEAREST_STOD = 0; NEAREST_IDAVG = 1; CREEP_FILL = 2
    class MeshLoc: NODE = 0; ELEMENT = 1
    class MeshElemType: TRI = 1; QUAD = 2
    class NormType: FRACAREA = 0; DSTAREA = 1
    class Manager:
        def __init__(self, *args, **kwargs): pass
    @staticmethod
    def pet_count(): return 1
    @staticmethod
    def local_pet(): return 0
    class Grid:
        def __init__(self, shape, *args, **kwargs):
            self.shape = shape
            self.coords = [np.zeros(shape), np.zeros(shape)]
            self.staggerloc = [0, 1]
        def get_coords(self, i, staggerloc=0): return self.coords[i]
        def add_item(self, *args, **kwargs): pass
        def get_item(self, *args, **kwargs): return np.zeros(self.shape)
    class LocStream:
        def __init__(self, *args, **kwargs): self.items = {}
        def __setitem__(self, k, v): self.items[k] = v
        def __getitem__(self, k): return self.items[k]
    class Mesh:
        def __init__(self, *args, **kwargs): pass
        def add_nodes(self, *args, **kwargs): pass
        def add_elements(self, *args, **kwargs): pass
    class Field:
        def __init__(self, *args, **kwargs): pass
    class Regrid:
        def __init__(self, src, dst, **kwargs):
            self.src_shape = src.shape # (n_lon_src, n_lat_src)
            self.dst_shape = dst.shape # (n_lon_dst, n_lat_dst)
        def get_factors(self, *args, **kwargs): return np.array([1]), np.array([1])
        def get_weights_dict(self, *args, **kwargs):
            # Identity weights for test
            n_src = self.src_shape[0] * self.src_shape[1]
            n_dst = self.dst_shape[0] * self.dst_shape[1]
            n = min(n_src, n_dst)
            return {
                "row_dst": np.arange(1, n + 1),
                "col_src": np.arange(1, n + 1),
                "weights": np.ones(n),
            }

sys.modules["esmpy"] = MockEsmpy
from xregrid import Regridder

def test_descending_lat():
    # Source grid (5x10) with descending latitude
    lat = np.linspace(50, 10, 5) # [50, 40, 30, 20, 10]
    lon = np.linspace(100, 190, 10) # [100, 110, ..., 190]

    data = np.arange(50).reshape(5, 10)
    # data[0, 0] is lat=50, lon=100. Index 0 in flattened.

    ds = xr.Dataset(
        {"air": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon}
    )

    # Target grid is same
    target_grid = ds.copy()

    regridder = Regridder(ds, target_grid, method="bilinear")
    regridded = regridder(ds.air)

    print("Source data at (lat=50, lon=100):", ds.air.sel(lat=50, lon=100).values)
    print("Regridded data at (lat=50, lon=100):", regridded.sel(lat=50, lon=100).values)

    np.testing.assert_allclose(ds.air.values, regridded.values)
    print("SUCCESS: Descending lat regridding passed!")

if __name__ == "__main__":
    test_descending_lat()
