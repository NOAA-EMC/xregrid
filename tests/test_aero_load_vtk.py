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


class TestLoadVtkDirectory:
    """Validate loading a directory of per-rank VTK files as one mesh."""

    def _write_split_mesh(self, tmp_path, n_parts=3, cells_per_part=5):
        """Write multiple VTK files that share boundary nodes."""
        mesh_dir = tmp_path / "mesh_parts"
        mesh_dir.mkdir()
        rng = np.random.default_rng(123)

        total_cells = 0
        for p in range(n_parts):
            n_points = cells_per_part * 3
            # Offset longitude by partition to simulate spatial decomposition
            lon_base = p * 30.0
            lons = rng.uniform(lon_base, lon_base + 30, n_points)
            lats = rng.uniform(-60, 60, n_points)

            lines = [
                "# vtk DataFile Version 3.0",
                f"Partition {p}",
                "ASCII",
                "DATASET UNSTRUCTURED_GRID",
                f"POINTS {n_points} double",
            ]
            for i in range(n_points):
                lines.append(f"{lons[i]:.8f} {lats[i]:.8f} 0.0")

            total_ints = cells_per_part * 4
            lines.append(f"CELLS {cells_per_part} {total_ints}")
            for i in range(cells_per_part):
                base = i * 3
                lines.append(f"3 {base} {base + 1} {base + 2}")

            lines.append(f"CELL_TYPES {cells_per_part}")
            lines.extend(["5"] * cells_per_part)

            vtk_path = mesh_dir / f"mesh_{p:04d}.vtk"
            with open(vtk_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            total_cells += cells_per_part

        return str(mesh_dir), total_cells

    def test_loads_directory(self, tmp_path):
        mesh_dir, expected_cells = self._write_split_mesh(tmp_path)
        ds = load_vtk_mesh(mesh_dir)
        assert isinstance(ds, xr.Dataset)
        assert ds.sizes["nFaces"] == expected_cells

    def test_merged_connectivity_valid(self, tmp_path):
        mesh_dir, _ = self._write_split_mesh(tmp_path)
        ds = load_vtk_mesh(mesh_dir)
        conn = ds["face_node_connectivity"].values
        n_nodes = ds.sizes["nNodes"]
        valid = conn[conn != -1]
        assert valid.min() >= 0
        assert valid.max() < n_nodes

    def test_history_mentions_merge(self, tmp_path):
        mesh_dir, _ = self._write_split_mesh(tmp_path, n_parts=4)
        ds = load_vtk_mesh(mesh_dir)
        assert "Merged 4 VTK files" in ds.attrs["history"]

    def test_ugrid_structure(self, tmp_path):
        mesh_dir, _ = self._write_split_mesh(tmp_path)
        ds = load_vtk_mesh(mesh_dir)
        assert ds["mesh"].attrs["cf_role"] == "mesh_topology"
        assert "node_lon" in ds.coords
        assert "node_lat" in ds.coords

    def test_empty_directory_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="no .vtk files"):
            load_vtk_mesh(str(empty_dir))

    def test_grid_detection_on_merged(self, tmp_path):
        """Merged dataset must be recognized as unstructured by xregrid."""
        mesh_dir, _ = self._write_split_mesh(tmp_path)
        ds = load_vtk_mesh(mesh_dir)
        _, _, _, _, is_unstructured = _get_mesh_info(ds)
        assert is_unstructured

    def test_node_deduplication(self, tmp_path):
        """Shared boundary nodes should be deduplicated."""
        mesh_dir = tmp_path / "shared_nodes"
        mesh_dir.mkdir()

        # Two files that share 2 nodes at the boundary
        shared_lon, shared_lat = 15.0, 30.0
        shared_lon2, shared_lat2 = 15.0, 31.0

        for idx, (extra_lon, extra_lat) in enumerate(
            [(10.0, 30.5), (20.0, 30.5)]
        ):
            lines = [
                "# vtk DataFile Version 3.0",
                f"Part {idx}",
                "ASCII",
                "DATASET UNSTRUCTURED_GRID",
                "POINTS 3 double",
                f"{shared_lon:.8f} {shared_lat:.8f} 0.0",
                f"{shared_lon2:.8f} {shared_lat2:.8f} 0.0",
                f"{extra_lon:.8f} {extra_lat:.8f} 0.0",
                "CELLS 1 4",
                "3 0 1 2",
                "CELL_TYPES 1",
                "5",
            ]
            with open(mesh_dir / f"part_{idx}.vtk", "w") as f:
                f.write("\n".join(lines) + "\n")

        ds = load_vtk_mesh(str(mesh_dir))
        # 6 raw nodes, but 2 are shared → 4 unique
        assert ds.sizes["nNodes"] == 4
        assert ds.sizes["nFaces"] == 2
