import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def create_mock_ugrid(n_nodes=10, n_faces=4):
    """Create a mock UGRID dataset."""
    # Nodes
    node_x = np.linspace(0, 360, n_nodes)
    node_y = np.linspace(-90, 90, n_nodes)

    # Faces (centers)
    face_x = np.linspace(10, 350, n_faces)
    face_y = np.linspace(-80, 80, n_faces)

    # Simple triangulation: face 0 uses nodes 0,1,2; face 1 uses 1,2,3...
    face_nodes = np.zeros((n_faces, 3), dtype=int)
    for i in range(n_faces):
        face_nodes[i] = [i, i + 1, (i + 2) % n_nodes]

    ds = xr.Dataset(
        data_vars={
            "mesh_topology": (
                [],
                0,
                {
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "node_lon node_lat",
                    "face_node_connectivity": "face_nodes",
                    "face_coordinates": "face_lon face_lat",
                },
            ),
            "face_nodes": (
                ["n_face", "n_node_per_face"],
                face_nodes,
                {"cf_role": "face_node_connectivity", "start_index": 0},
            ),
            "temp": (
                ["n_face"],
                np.random.rand(n_faces),
                {
                    "mesh": "mesh_topology",
                    "location": "face",
                    "standard_name": "air_temperature",
                },
            ),
        },
        coords={
            "node_lon": (
                ["n_node"],
                node_x,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "node_lat": (
                ["n_node"],
                node_y,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "face_lon": (
                ["n_face"],
                face_x,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "face_lat": (
                ["n_face"],
                face_y,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
    )
    return ds


def test_ugrid_discovery_and_regrid():
    """Verify UGRID discovery and regridding (Eager and Lazy)."""
    src_ds = create_mock_ugrid(n_nodes=20, n_faces=10)
    tgt_grid = create_global_grid(30, 30)

    # Test that Regridder can handle the UGRID dataset
    # We use conservative regridding to test triangulation logic
    regridder = Regridder(src_ds, tgt_grid, method="conservative")

    # 1. Eager test
    res_eager = regridder(src_ds.temp)

    assert "lat" in res_eager.coords
    assert "lon" in res_eager.coords
    # Verify metadata removal as target is not UGRID
    assert "mesh" not in res_eager.attrs

    # 2. Lazy test
    da_lazy = src_ds.temp.chunk({"n_face": 5})
    res_lazy = regridder(da_lazy).compute()

    xr.testing.assert_allclose(res_eager, res_lazy)


def test_ugrid_scientific_hygiene():
    """Verify UGRID metadata propagation to UGRID target."""
    src_ds = create_mock_ugrid(n_nodes=10, n_faces=4)
    tgt_ds = create_mock_ugrid(n_nodes=15, n_faces=6)

    # For simplicity, use nearest_s2d which doesn't require complex connectivity for target
    regridder = Regridder(src_ds, tgt_ds, method="nearest_s2d")

    res = regridder(src_ds.temp)

    # Scientific Hygiene: target mesh should be attached
    assert res.attrs["mesh"] == "mesh_topology"
    assert "location" in res.attrs
    assert "mesh_topology" in res.coords or "mesh_topology" in res.data_vars


if __name__ == "__main__":
    pytest.main([__file__])
