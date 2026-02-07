import sys
from unittest.mock import MagicMock
import numpy as np

try:
    import esmpy  # noqa: F401
except ImportError:
    mock_esmpy = MagicMock()
    mock_esmpy.CoordSys.SPH_DEG = 1
    mock_esmpy.StaggerLoc.CENTER = 0
    mock_esmpy.StaggerLoc.CORNER = 1
    mock_esmpy.GridItem.MASK = 1
    mock_esmpy.RegridMethod.BILINEAR = 0
    mock_esmpy.RegridMethod.CONSERVE = 1
    mock_esmpy.RegridMethod.NEAREST_STOD = 2
    mock_esmpy.RegridMethod.NEAREST_DTOS = 3
    mock_esmpy.RegridMethod.PATCH = 4
    mock_esmpy.UnmappedAction.IGNORE = 1
    mock_esmpy.ExtrapMethod.NEAREST_STOD = 0
    mock_esmpy.ExtrapMethod.NEAREST_IDAVG = 1
    mock_esmpy.ExtrapMethod.CREEP_FILL = 2
    mock_esmpy.MeshLoc.NODE = 0
    mock_esmpy.MeshLoc.ELEMENT = 1
    mock_esmpy.MeshElemType.TRI = 1
    mock_esmpy.MeshElemType.QUAD = 2
    mock_esmpy.NormType.FRACAREA = 0
    mock_esmpy.NormType.DSTAREA = 1

    # Mock Manager
    mock_esmpy.Manager.return_value = MagicMock()
    mock_esmpy.pet_count.return_value = 1
    mock_esmpy.local_pet.return_value = 0

    # Mock Grid
    class Grid:
        def __init__(self, *args, **kwargs):
            self.get_coords = MagicMock()
            self.get_item = MagicMock()
            self.add_item = MagicMock()
            self.staggerloc = [0, 1]

    mock_esmpy.Grid = Grid

    class LocStream:
        def __init__(self, *args, **kwargs):
            self.items = {}

        def __setitem__(self, key, value):
            self.items[key] = value

    mock_esmpy.LocStream = LocStream

    class Mesh:
        def __init__(self, *args, **kwargs):
            self.nodes = []
            self.elements = []

        def add_nodes(self, *args, **kwargs):
            pass

        def add_elements(self, *args, **kwargs):
            pass

    mock_esmpy.Mesh = Mesh

    # Mock Field
    mock_esmpy.Field.return_value = MagicMock()

    # Mock Regrid
    mock_regrid = MagicMock()
    mock_regrid.get_factors.return_value = (np.array([0]), np.array([0]))
    # Mock weights for a small 10x20 -> 15x25 regrid
    # n_src = 200, n_dst = 375
    mock_regrid.get_weights_dict.return_value = {
        "row_dst": np.array([1]),
        "col_src": np.array([1]),
        "weights": np.array([1.0]),
    }
    mock_esmpy.Regrid.return_value = mock_regrid

    sys.modules["esmpy"] = mock_esmpy
