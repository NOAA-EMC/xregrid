
import sys
import numpy as np
import xarray as xr

class MockEsmpy:
    class CoordSys:
        SPH_DEG = 1
    class StaggerLoc:
        CENTER = 0
        CORNER = 1
    class GridItem:
        MASK = 1
    class RegridMethod:
        BILINEAR = 0
        CONSERVE = 1
        NEAREST_STOD = 2
        NEAREST_DTOS = 3
        PATCH = 4
    class UnmappedAction:
        IGNORE = 1
    class ExtrapMethod:
        NEAREST_STOD = 0
        NEAREST_IDAVG = 1
        CREEP_FILL = 2
    class MeshLoc:
        NODE = 0
        ELEMENT = 1
    class MeshElemType:
        TRI = 1
        QUAD = 2
    class NormType:
        FRACAREA = 0
        DSTAREA = 1
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
        def get_coords(self, i, staggerloc=0):
            return self.coords[i]
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
        def __init__(self, *args, **kwargs): pass
        def get_factors(self, *args, **kwargs):
            return np.array([1]), np.array([1])
        def get_weights_dict(self, *args, **kwargs):
            # We'll set this dynamically in the test
            return {}

sys.modules["esmpy"] = MockEsmpy
from xregrid import Regridder

def test_identity():
    # Create a 10x10 source grid
    lat = np.arange(10)
    lon = np.arange(10)
    ds = xr.Dataset(
        {"air": (["lat", "lon"], np.random.rand(10, 10))},
        coords={"lat": lat, "lon": lon}
    )

    # Target grid is the same
    target_grid = ds.copy()

    # Setup mock weights for identity
    n_src = 100
    MockEsmpy.Regrid.get_weights_dict = lambda self, **kwargs: {
        "row_dst": np.arange(1, n_src + 1),
        "col_src": np.arange(1, n_src + 1),
        "weights": np.ones(n_src),
    }

    regridder = Regridder(ds, target_grid, method="bilinear")
    regridded = regridder(ds.air)

    print("Source values (first 2x2):\n", ds.air.values[:2, :2])
    print("Regridded values (first 2x2):\n", regridded.values[:2, :2])

    np.testing.assert_allclose(ds.air.values, regridded.values)
    print("SUCCESS: Identity regridding passed!")

if __name__ == "__main__":
    test_identity()
