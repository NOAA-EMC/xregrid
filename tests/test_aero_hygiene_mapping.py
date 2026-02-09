import sys
from unittest.mock import MagicMock

import numpy as np

# Mock esmpy BEFORE importing xregrid to allow running in environments without ESMF
try:
    import esmpy  # noqa: F401

    ESMPY_AVAILABLE = True
except ImportError:
    ESMPY_AVAILABLE = False

if not ESMPY_AVAILABLE:
    mock_esmpy = MagicMock()
    mock_esmpy.CoordSys.SPH_DEG = 1
    mock_esmpy.CoordSys.CART = 0
    mock_esmpy.StaggerLoc.CENTER = 0
    mock_esmpy.StaggerLoc.CORNER = 1
    mock_esmpy.RegridMethod.BILINEAR = 0
    mock_esmpy.UnmappedAction.IGNORE = 1

    class MockESMFObject:
        pass

    class Grid(MockESMFObject):
        def __init__(self, shape, *args, **kwargs):
            self.shape = shape

        def get_coords(self, dim, staggerloc=None):
            return np.zeros(self.shape)

        def add_item(self, *args, **kwargs):
            pass

        def get_item(self, *args, **kwargs):
            return np.zeros(self.shape)

    class Mesh(MockESMFObject):
        def __init__(self, *args, **kwargs):
            pass

        def add_nodes(self, *args, **kwargs):
            pass

        def add_elements(self, *args, **kwargs):
            pass

    class LocStream(MockESMFObject):
        def __init__(self, size, *args, **kwargs):
            self.size = size

        def __getitem__(self, key):
            return np.zeros(self.size)

        def __setitem__(self, key, value):
            pass

    mock_esmpy.Grid = Grid
    mock_esmpy.Mesh = Mesh
    mock_esmpy.LocStream = LocStream
    mock_esmpy.Field = MagicMock()

    class MockRegrid:
        def __init__(self, *args, **kwargs):
            pass

        def get_weights_dict(self, deep_copy=True):
            return {
                "row_dst": np.array([1]),
                "col_src": np.array([1]),
                "weights": np.array([1.0]),
            }

        def get_factors(self):
            return np.array([1.0]), np.array([1])

    mock_esmpy.Regrid = MockRegrid
    mock_esmpy.pet_count.return_value = 1
    mock_esmpy.local_pet.return_value = 0
    sys.modules["esmpy"] = mock_esmpy

import xarray as xr  # noqa: E402
from xregrid import Regridder  # noqa: E402


def test_grid_mapping_hygiene():
    """Verify that grid_mapping is correctly updated and preserved (Aero Protocol)."""

    # 1. Setup source grid with grid_mapping
    src_ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(10)),
            "lon": (["lon"], np.arange(10)),
        }
    )
    src_ds["crs_src"] = xr.DataArray(
        0, attrs={"grid_mapping_name": "latitude_longitude"}
    )

    da_src = xr.DataArray(
        np.random.rand(10, 10),
        dims=("lat", "lon"),
        coords=src_ds.coords,
        name="test_data",
        attrs={"grid_mapping": "crs_src"},
    )

    # 2. Setup target grid with different grid_mapping
    tgt_ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(5)),
            "lon": (["lon"], np.arange(5)),
        }
    )
    tgt_ds["crs_tgt"] = xr.DataArray(0, attrs={"grid_mapping_name": "mercator"})

    regridder = Regridder(src_ds, tgt_ds, method="bilinear")

    # --- Test Track A: Eager (NumPy) ---
    out_eager = regridder(da_src)

    assert out_eager.attrs["grid_mapping"] == "crs_tgt"
    assert "crs_tgt" in out_eager.coords
    assert "crs_src" not in out_eager.coords

    # --- Test Track B: Lazy (Dask) ---
    da_lazy = da_src.chunk({"lat": 5, "lon": 5})
    out_lazy = regridder(da_lazy)

    assert out_lazy.attrs["grid_mapping"] == "crs_tgt"
    assert "crs_tgt" in out_lazy.coords

    # Verify that the logic holds after computation
    out_computed = out_lazy.compute()
    assert out_computed.attrs["grid_mapping"] == "crs_tgt"
    assert "crs_tgt" in out_computed.coords


def test_dataset_grid_mapping_hygiene():
    """Verify that grid_mapping is correctly updated for Datasets."""

    src_ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(10)),
            "lon": (["lon"], np.arange(10)),
        }
    )
    src_ds["crs_src"] = xr.DataArray(
        0, attrs={"grid_mapping_name": "latitude_longitude"}
    )
    src_ds["var1"] = (["lat", "lon"], np.random.rand(10, 10))
    src_ds["var1"].attrs["grid_mapping"] = "crs_src"
    src_ds.attrs["grid_mapping"] = "crs_src"

    tgt_ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(5)),
            "lon": (["lon"], np.arange(5)),
        }
    )
    tgt_ds["crs_tgt"] = xr.DataArray(0, attrs={"grid_mapping_name": "mercator"})

    regridder = Regridder(src_ds, tgt_ds)
    out_ds = regridder(src_ds)

    assert out_ds.attrs["grid_mapping"] == "crs_tgt"
    assert out_ds["var1"].attrs["grid_mapping"] == "crs_tgt"
    assert "crs_tgt" in out_ds.coords
    assert "crs_src" not in out_ds.coords
