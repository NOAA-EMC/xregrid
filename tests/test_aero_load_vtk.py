"""Tests for load_vtk_mesh: VTK legacy file → UGRID xr.Dataset."""

import numpy as np
import pytest
import xarray as xr

from xregrid.utils import load_vtk_mesh
from xregrid.grid import _get_mesh_info, _get_unstructured_mesh_info


def _write_vtk_file(path: str, n_cells: int = 20, verts_per_cell: int = 3):
    """Write a synthetic VTK legacy unstructured grid file."""
    rng = np.random.default_rng(42)
    n_points = n_cells * verts_per_cell
    lons = rng.uniform(10, 350, n_points)
    lats = rng.uniform(-80, 80, n_points)

    vtk_type = {3: 5, 4: 9}.get(verts_per_cell, 7)  # TRI, QUAD, or POLYGON

    lines = [
        "# vtk DataFile Version 3.0",
        "Test mesh",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} double",
    ]
    for i in range(n_points):
        lines.append(f"{lons[i]:.8f} {lats[i]:.8f} 0.0")

    total_ints = n_cells * (verts_per_cell + 1)
    lines.append(f"CELLS {n_cells} {total_ints}")
    for i in range(n_cells):
        base = i * verts_per_cell
        idx = " ".join(str(base + j) for j in range(verts_per_cell))
        lines.append(f"{verts_per_cell} {idx}")

    lines.append(f"CELL_TYPES {n_cells}")
    for _ in range(n_cells):
        lines.append(str(vtk_type))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return lons, lats


class TestLoadVtkMesh:
    """Validate load_vtk_mesh produces a well-formed UGRID dataset."""

    def test_returns_dataset(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk)
        ds = load_vtk_mesh(vtk)
        assert isinstance(ds, xr.Dataset)

    def test_ugrid_variables_present(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk)
        ds = load_vtk_mesh(vtk)
        assert "mesh" in ds
        assert "face_node_connectivity" in ds
        assert "node_lon" in ds.coords
        assert "node_lat" in ds.coords

    def test_mesh_topology_attrs(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk)
        ds = load_vtk_mesh(vtk)
        assert ds["mesh"].attrs["cf_role"] == "mesh_topology"
        assert ds["mesh"].attrs["topology_dimension"] == 2

    def test_connectivity_shape(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        n_cells, verts = 15, 3
        _write_vtk_file(vtk, n_cells=n_cells, verts_per_cell=verts)
        ds = load_vtk_mesh(vtk)
        conn = ds["face_node_connectivity"].values
        assert conn.shape == (n_cells, verts)

    def test_connectivity_attrs(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk)
        ds = load_vtk_mesh(vtk)
        attrs = ds["face_node_connectivity"].attrs
        assert attrs["start_index"] == 0
        assert attrs["_FillValue"] == -1
        assert attrs["cf_role"] == "face_node_connectivity"

    def test_node_coords_match_vtk(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        lons, lats = _write_vtk_file(vtk, n_cells=5, verts_per_cell=3)
        ds = load_vtk_mesh(vtk)
        np.testing.assert_allclose(ds["node_lon"].values, lons)
        np.testing.assert_allclose(ds["node_lat"].values, lats)

    def test_cf_attributes(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk)
        ds = load_vtk_mesh(vtk)
        assert ds["node_lon"].attrs["standard_name"] == "longitude"
        assert ds["node_lat"].attrs["standard_name"] == "latitude"

    def test_history_updated(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk)
        ds = load_vtk_mesh(vtk)
        assert "history" in ds.attrs
        assert "VTK" in ds.attrs["history"]

    def test_quad_cells(self, tmp_path):
        vtk = str(tmp_path / "quad.vtk")
        _write_vtk_file(vtk, n_cells=10, verts_per_cell=4)
        ds = load_vtk_mesh(vtk)
        assert ds["face_node_connectivity"].shape == (10, 4)

    def test_invalid_file_raises(self, tmp_path):
        bad = str(tmp_path / "bad.vtk")
        with open(bad, "w") as f:
            f.write("garbage\n")
        with pytest.raises(ValueError, match="Could not parse"):
            load_vtk_mesh(bad)


class TestVtkGridDetection:
    """Validate that the UGRID dataset from load_vtk_mesh is recognized
    by xregrid's grid detection pipeline."""

    def test_detected_as_unstructured(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk, n_cells=10, verts_per_cell=3)
        ds = load_vtk_mesh(vtk)
        lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
        assert is_unstructured

    def test_mesh_info_extraction(self, tmp_path):
        vtk = str(tmp_path / "mesh.vtk")
        _write_vtk_file(vtk, n_cells=10, verts_per_cell=3)
        ds = load_vtk_mesh(vtk)
        node_lon, node_lat, elem_conn, elem_types, elem_ids, orig_idx = (
            _get_unstructured_mesh_info(ds, method="conservative")
        )
        assert len(node_lon) == 30  # 10 cells * 3 verts
        assert len(elem_ids) > 0
        assert elem_conn.min() >= 0
        assert elem_conn.max() < len(node_lon)
