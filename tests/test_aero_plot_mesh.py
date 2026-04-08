"""Tests for plot_mesh: unstructured mesh wireframe visualization."""

import numpy as np
import pytest
import xarray as xr

from xregrid.viz import plot_mesh, _extract_cell_polygons

plt = pytest.importorskip("matplotlib.pyplot")
ccrs = pytest.importorskip("cartopy.crs")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mpas_mesh(n_cells: int = 50, max_edges: int = 6) -> xr.Dataset:
    """Create a synthetic MPAS-style mesh dataset."""
    rng = np.random.default_rng(42)

    # Random cell centres
    cell_lat = rng.uniform(-80, 80, n_cells)
    cell_lon = rng.uniform(0, 360, n_cells)

    # Build fake vertices around each cell centre
    n_vertices = n_cells * max_edges
    vert_lat = np.zeros(n_vertices)
    vert_lon = np.zeros(n_vertices)
    conn = np.zeros((n_cells, max_edges), dtype=int)
    n_edges_on_cell = np.full(n_cells, max_edges, dtype=int)

    for i in range(n_cells):
        angles = np.linspace(0, 2 * np.pi, max_edges, endpoint=False)
        radius = 2.0
        base = i * max_edges
        vert_lat[base : base + max_edges] = cell_lat[i] + radius * np.sin(angles)
        vert_lon[base : base + max_edges] = cell_lon[i] + radius * np.cos(angles)
        conn[i] = np.arange(base, base + max_edges) + 1  # 1-based

    return xr.Dataset(
        {
            "latVertex": (["nVertices"], np.radians(vert_lat)),
            "lonVertex": (["nVertices"], np.radians(vert_lon)),
            "verticesOnCell": (["nCells", "maxEdges"], conn),
            "nEdgesOnCell": (["nCells"], n_edges_on_cell),
        }
    )


def _make_ugrid_mesh(n_cells: int = 30, n_verts_per_face: int = 4) -> xr.Dataset:
    """Create a synthetic UGRID-style mesh dataset."""
    rng = np.random.default_rng(99)
    n_nodes = n_cells * n_verts_per_face
    node_lon = rng.uniform(0, 360, n_nodes).astype(np.float64)
    node_lat = rng.uniform(-90, 90, n_nodes).astype(np.float64)
    conn = np.arange(n_nodes, dtype=int).reshape(n_cells, n_verts_per_face)

    return xr.Dataset(
        {
            "mesh": (
                [],
                0,
                {
                    "cf_role": "mesh_topology",
                    "node_coordinates": "node_lon node_lat",
                    "face_node_connectivity": "face_node_connectivity",
                },
            ),
            "node_lon": (["nNodes"], node_lon, {"standard_name": "longitude"}),
            "node_lat": (["nNodes"], node_lat, {"standard_name": "latitude"}),
            "face_node_connectivity": (
                ["nFaces", "nMaxVerts"],
                conn,
                {"cf_role": "face_node_connectivity", "start_index": 0},
            ),
        }
    )


def _make_scrip_mesh(n_cells: int = 20, n_corners: int = 4) -> xr.Dataset:
    """Create a synthetic SCRIP-style mesh dataset."""
    rng = np.random.default_rng(7)
    lat_b = rng.uniform(-80, 80, (n_cells, n_corners))
    lon_b = rng.uniform(0, 360, (n_cells, n_corners))
    return xr.Dataset(
        {
            "lat_b": (["nCells", "nCorners"], lat_b),
            "lon_b": (["nCells", "nCorners"], lon_b),
        }
    )


# ---------------------------------------------------------------------------
# Tests: _extract_cell_polygons
# ---------------------------------------------------------------------------


class TestExtractCellPolygons:
    """Unit tests for polygon extraction from different mesh conventions."""

    def test_mpas_polygons(self):
        ds = _make_mpas_mesh(n_cells=10, max_edges=5)
        polys = _extract_cell_polygons(ds)
        assert len(polys) == 10
        for p in polys:
            assert p.shape == (5, 2)

    def test_ugrid_polygons(self):
        ds = _make_ugrid_mesh(n_cells=8, n_verts_per_face=3)
        polys = _extract_cell_polygons(ds)
        assert len(polys) == 8
        for p in polys:
            assert p.shape == (3, 2)

    def test_scrip_polygons(self):
        ds = _make_scrip_mesh(n_cells=5, n_corners=4)
        polys = _extract_cell_polygons(ds)
        assert len(polys) == 5
        for p in polys:
            assert p.shape == (4, 2)

    def test_unsupported_raises(self):
        ds = xr.Dataset({"temperature": (["x"], [1, 2, 3])})
        with pytest.raises(ValueError, match="Could not detect"):
            _extract_cell_polygons(ds)


# ---------------------------------------------------------------------------
# Tests: plot_mesh (integration with matplotlib + cartopy)
# ---------------------------------------------------------------------------


class TestPlotMesh:
    """Integration tests for the plot_mesh function."""

    def test_mpas_default(self):
        ds = _make_mpas_mesh()
        coll = plot_mesh(ds, title="MPAS test")
        assert coll is not None
        plt.close("all")

    def test_ugrid_custom_projection(self):
        ds = _make_ugrid_mesh()
        coll = plot_mesh(
            ds,
            projection=ccrs.Robinson(),
            edgecolor="blue",
            linewidth=0.5,
        )
        assert coll is not None
        plt.close("all")

    def test_scrip_mesh(self):
        ds = _make_scrip_mesh()
        coll = plot_mesh(ds)
        assert coll is not None
        plt.close("all")

    def test_existing_axes(self):
        ds = _make_mpas_mesh(n_cells=10)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        coll = plot_mesh(ds, ax=ax)
        assert coll is not None
        plt.close("all")

    def test_facecolor_filled(self):
        ds = _make_mpas_mesh(n_cells=10)
        coll = plot_mesh(ds, facecolor="lightblue", alpha=0.5)
        assert coll is not None
        plt.close("all")


from xregrid.viz import _read_vtk_polygons


# ---------------------------------------------------------------------------
# VTK legacy format support
# ---------------------------------------------------------------------------


def _write_vtk_triangles(path: str, n_cells: int = 10):
    """Write a minimal VTK legacy file with triangular cells."""
    rng = np.random.default_rng(77)
    n_points = n_cells * 3
    lons = rng.uniform(0, 360, n_points)
    lats = rng.uniform(-80, 80, n_points)

    lines = [
        "# vtk DataFile Version 3.0",
        "Test ESMF Mesh",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} double",
    ]
    for i in range(n_points):
        lines.append(f"{lons[i]:.6f} {lats[i]:.6f} 0.0")

    total_ints = n_cells * 4  # each cell: 3 + count prefix
    lines.append(f"CELLS {n_cells} {total_ints}")
    for i in range(n_cells):
        base = i * 3
        lines.append(f"3 {base} {base + 1} {base + 2}")

    lines.append(f"CELL_TYPES {n_cells}")
    for _ in range(n_cells):
        lines.append("5")  # VTK_TRIANGLE

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_vtk_quads(path: str, n_cells: int = 8):
    """Write a minimal VTK legacy file with quad cells."""
    rng = np.random.default_rng(88)
    n_points = n_cells * 4
    lons = rng.uniform(0, 360, n_points)
    lats = rng.uniform(-80, 80, n_points)

    lines = [
        "# vtk DataFile Version 3.0",
        "Test ESMF Quad Mesh",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} double",
    ]
    for i in range(n_points):
        lines.append(f"{lons[i]:.6f} {lats[i]:.6f} 0.0")

    total_ints = n_cells * 5
    lines.append(f"CELLS {n_cells} {total_ints}")
    for i in range(n_cells):
        base = i * 4
        lines.append(f"4 {base} {base + 1} {base + 2} {base + 3}")

    lines.append(f"CELL_TYPES {n_cells}")
    for _ in range(n_cells):
        lines.append("9")  # VTK_QUAD

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_vtk_mixed(path: str):
    """Write a VTK file with mixed tri + quad + polygon cells."""
    lines = [
        "# vtk DataFile Version 3.0",
        "Mixed mesh",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "POINTS 9 double",
        "0.0 0.0 0.0",
        "10.0 0.0 0.0",
        "5.0 10.0 0.0",
        "20.0 0.0 0.0",
        "30.0 0.0 0.0",
        "30.0 10.0 0.0",
        "20.0 10.0 0.0",
        "40.0 5.0 0.0",
        "35.0 15.0 0.0",
        "CELLS 3 15",
        "3 0 1 2",       # triangle
        "4 3 4 5 6",     # quad
        "5 4 7 8 5 6",   # polygon (pentagon) — VTK type 7 is not used here
        "CELL_TYPES 3",
        "5",   # TRI
        "9",   # QUAD
        "7",   # POLYGON
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class TestReadVtkPolygons:
    """Unit tests for VTK legacy file parsing."""

    def test_triangles(self, tmp_path):
        vtk_file = str(tmp_path / "tri.vtk")
        _write_vtk_triangles(vtk_file, n_cells=10)
        polys = _read_vtk_polygons(vtk_file)
        assert len(polys) == 10
        for p in polys:
            assert p.shape == (3, 2)

    def test_quads(self, tmp_path):
        vtk_file = str(tmp_path / "quad.vtk")
        _write_vtk_quads(vtk_file, n_cells=8)
        polys = _read_vtk_polygons(vtk_file)
        assert len(polys) == 8
        for p in polys:
            assert p.shape == (4, 2)

    def test_mixed(self, tmp_path):
        vtk_file = str(tmp_path / "mixed.vtk")
        _write_vtk_mixed(vtk_file)
        polys = _read_vtk_polygons(vtk_file)
        assert len(polys) == 3
        assert polys[0].shape == (3, 2)  # tri
        assert polys[1].shape == (4, 2)  # quad
        assert polys[2].shape == (5, 2)  # polygon

    def test_invalid_file_raises(self, tmp_path):
        bad_file = str(tmp_path / "bad.vtk")
        with open(bad_file, "w") as f:
            f.write("not a vtk file\n")
        with pytest.raises(ValueError, match="Could not parse"):
            _read_vtk_polygons(bad_file)


class TestPlotMeshVtk:
    """Integration tests for plot_mesh with VTK file paths."""

    def test_plot_vtk_triangles(self, tmp_path):
        vtk_file = str(tmp_path / "tri.vtk")
        _write_vtk_triangles(vtk_file)
        coll = plot_mesh(vtk_file, title="VTK Triangles")
        assert coll is not None
        plt.close("all")

    def test_plot_vtk_quads(self, tmp_path):
        vtk_file = str(tmp_path / "quad.vtk")
        _write_vtk_quads(vtk_file)
        coll = plot_mesh(vtk_file, projection=ccrs.Robinson())
        assert coll is not None
        plt.close("all")
