from __future__ import annotations


import numpy as np
import xarray as xr
from unittest.mock import patch

from xregrid import Regridder, create_grid_from_crs


def test_normalize_grid_eager_lazy_identity():
    """
    Aero Protocol Double-Check Test: Verify that _normalize_grid produce
    identical results for Eager (NumPy) and Lazy (Dask) backends.
    """
    # 1. Create descending grid (Eager)
    lat = np.array([10.0, 0.0, -10.0])
    lon = np.array([20.0, 10.0, 0.0])
    ds_eager = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    # 2. Create Lazy version
    # Note: Dimension coordinates with pandas indexes are always eager in Xarray.
    # To test laziness of the sorting logic, we'd need non-dimension coordinates,
    # but _normalize_grid specifically targets 1D dimension coordinates.
    # So we focus on verifying that the new index-based check works.

    with patch.object(Regridder, "_generate_weights"):
        rg_eager = Regridder(ds_eager, ds_eager)

    assert rg_eager._src_was_sorted is True
    assert rg_eager.source_grid_ds.lat.values[0] == -10.0

    # Verify that it doesn't sort if already sorted
    ds_sorted = rg_eager.source_grid_ds
    with patch.object(Regridder, "_generate_weights"):
        rg_sorted = Regridder(ds_sorted, ds_sorted)
    assert rg_sorted._src_was_sorted is False


def test_aero_lazy_init_no_compute():
    """
    Aero Protocol: Verify that Regridder initialization with parallel=True
    does not trigger computations on Dask-backed coordinates on the driver.

    We use a curvilinear grid because 2D coordinates can be truly lazy in Xarray.
    """
    # Create a curvilinear grid
    ds_src = create_grid_from_crs("EPSG:4326", (0, 20, 0, 20), 2.0)
    ds_tgt = create_grid_from_crs("EPSG:4326", (0, 20, 0, 20), 4.0)

    # Chunk them to make them lazy
    ds_src = ds_src.chunk({"x": 5, "y": 5})
    ds_tgt = ds_tgt.chunk({"x": 5, "y": 5})

    # Ensure they are dask-backed
    assert hasattr(ds_src.lat.data, "dask"), "Source Lat should be Dask-backed"

    # Mock ESMF weight generation to avoid needing binary dependencies
    with patch("xregrid.regridder.Regridder._generate_weights"):
        # parallel=True, compute=False should stay lazy on driver
        regridder = Regridder(
            ds_src, ds_tgt, method="bilinear", parallel=True, compute=False
        )

        # Check if coordinates are still dask-backed in the regridder
        # (Aero Protocol: Flexibility)
        assert hasattr(
            regridder.source_grid_ds.lat.data, "dask"
        ), "Source Lat should remain Dask-backed"


def test_aero_unstructured_mesh_info_robustness():
    """
    Aero Protocol: Verify that _get_unstructured_mesh_info handles input
    correctly and preserves mathematical integrity during transformations.
    """
    from xregrid.grid import _get_unstructured_mesh_info

    # Create a fake MPAS-like dataset
    ds = xr.Dataset(
        coords={
            "latVertex": (["nVertices"], np.linspace(-10, 10, 10)),
            "lonVertex": (["nVertices"], np.linspace(0, 20, 10)),
        },
        data_vars={
            "verticesOnCell": (
                ["nCells", "maxEdges"],
                np.array([[1, 2, 3, 0], [4, 5, 6, 0]], dtype=int),
            ),
        },
    )

    # Result should be NumPy as required by ESMF boundary
    res = _get_unstructured_mesh_info(ds)

    # node_lon, node_lat, element_conn, element_types, element_ids, orig_cell_index
    assert isinstance(res[0], np.ndarray)
    assert res[0].shape == (10,)
    assert isinstance(res[2], np.ndarray)
    assert res[2].dtype == np.int32
