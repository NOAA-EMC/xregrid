import pytest
import xarray as xr
import numpy as np
from xregrid.utils import _find_coord


def test_find_coord_ufs_names():
    """
    Verify that _find_coord correctly identifies UFS-style coordinate names.
    """
    # 1. Center coordinates (grid_latt, grid_lont)
    ds_center = xr.Dataset(
        coords={
            "grid_latt": (["y", "x"], np.zeros((10, 10))),
            "grid_lont": (["y", "x"], np.zeros((10, 10))),
        }
    )

    lat_da = _find_coord(ds_center, "latitude")
    lon_da = _find_coord(ds_center, "longitude")

    assert lat_da.name == "grid_latt"
    assert lon_da.name == "grid_lont"

    # 2. Corner coordinates (grid_lat, grid_lon)
    ds_corner = xr.Dataset(
        coords={
            "grid_lat": (["y_b", "x_b"], np.zeros((11, 11))),
            "grid_lon": (["y_b", "x_b"], np.zeros((11, 11))),
        }
    )

    lat_da_b = _find_coord(ds_corner, "latitude")
    lon_da_b = _find_coord(ds_corner, "longitude")

    assert lat_da_b.name == "grid_lat"
    assert lon_da_b.name == "grid_lon"


if __name__ == "__main__":
    pytest.main([__file__])
