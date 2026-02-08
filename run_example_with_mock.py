
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
            # Try to guess shapes
            self.n_src = 25 * 53 # matches air_temperature
            self.n_dst = 61 * 131 # matches target_grid
        def get_factors(self, *args, **kwargs): return np.array([1]), np.array([1])
        def get_weights_dict(self, *args, **kwargs):
            # For this test, let's just return SOME weights
            # Identity doesn't work because shapes differ.
            # Let's just map index k in src to index k in dst if possible,
            # or just map everything to index 0 of src.
            n = min(self.n_src, self.n_dst)
            return {
                "row_dst": np.arange(1, n + 1),
                "col_src": np.arange(1, n + 1),
                "weights": np.ones(n),
            }

sys.modules["esmpy"] = MockEsmpy

# Now run the example
import matplotlib
matplotlib.use('Agg') # No GUI
import docs.examples.scripts.plot_esmpy_comparison
