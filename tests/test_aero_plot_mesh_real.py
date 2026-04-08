"""Tests for plot_mesh using the real MPAS x1.2562 mesh file.

Downloads the mesh from UCAR, validates polygon extraction, the arrays
fed to ESMF Mesh creation, and the plot_mesh visualization.
"""

import os
import platform
import tarfile
import urllib.request

import numpy as np
import pytest
import xarray as xr

plt = pytest.importorskip("matplotlib.pyplot")
ccrs = pytest.importorskip("cartopy.crs")

import esmpy

from xregrid.grid import (
    _create_esmf_grid,
    _get_unstructured_mesh_info,
    _to_degrees,
    _clip_latitudes,
    _normalize_longitudes,
)
from xregrid.viz import _extract_cell_polygons, plot_mesh

# Detect whether we have real or mocked ESMF
HAS_REAL_ESMF = not getattr(esmpy, "_is_mock", False)
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# ---------------------------------------------------------------------------
# Fixture: download + cache the real MPAS mesh
# ---------------------------------------------------------------------------

MESH_URL = (
    "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes/x1.2562.tar.gz"
)
MESH_DIR = os.path.join(os.path.dirname(__file__), "_data")
MESH_NC = os.path.join(MESH_DIR, "x1.2562.grid.nc")


@pytest.fixture(scope="module")
def mpas_ds() -> xr.Dataset:
    """
    Download and open the MPAS x1.2562 mesh.

    Returns
    -------
    xr.Dataset
        The MPAS mesh dataset.
    """
    if not os.path.exists(MESH_NC):
        os.makedirs(MESH_DIR, exist_ok=True)
        tar_path = os.path.join(MESH_DIR, "x1.2562.tar.gz")
        urllib.request.urlretrieve(MESH_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extract("x1.2562.grid.nc", path=MESH_DIR)
        os.remove(tar_path)

    return xr.open_dataset(MESH_NC)


# ---------------------------------------------------------------------------
# 1. Polygon extraction from the real MPAS mesh
# ---------------------------------------------------------------------------


class TestRealMPASPolygons:
    """Validate _extract_cell_polygons against the real x1.2562 mesh."""

    def test_polygon_count_matches_cells(self, mpas_ds: xr.Dataset):
        """Each MPAS cell should produce exactly one polygon."""
        polys = _extract_cell_polygons(mpas_ds)
        n_cells = mpas_ds.sizes["nCells"]
        assert len(polys) == n_cells

    def test_polygon_vertex_counts(self, mpas_ds: xr.Dataset):
        """Vertex count per polygon must match nEdgesOnCell."""
        polys = _extract_cell_polygons(mpas_ds)
        n_edges = mpas_ds["nEdgesOnCell"].values
        for i, poly in enumerate(polys):
            assert poly.shape == (n_edges[i], 2), (
                f"Cell {i}: expected {n_edges[i]} vertices, got {poly.shape[0]}"
            )

    def test_coordinates_in_valid_range(self, mpas_ds: xr.Dataset):
        """All polygon vertices must be in valid geographic ranges."""
        polys = _extract_cell_polygons(mpas_ds)
        all_verts = np.concatenate(polys, axis=0)
        lons = all_verts[:, 0]
        lats = all_verts[:, 1]
        # Latitudes must be in [-90, 90]
        assert np.all(lats >= -90) and np.all(lats <= 90)
        # Longitudes may be shifted for dateline cells, but stay in [0, 720)
        assert np.all(lons >= 0) and np.all(lons < 720)


# ---------------------------------------------------------------------------
# 2. ESMF Mesh input arrays from the same file
# ---------------------------------------------------------------------------


class TestRealMPASEsmfMeshArrays:
    """Validate the arrays that _get_unstructured_mesh_info produces
    for the ESMF Mesh constructor."""

    def test_node_counts(self, mpas_ds: xr.Dataset):
        """Node arrays must match the MPAS vertex count."""
        node_lon, node_lat, _, _, _, _ = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        n_vertices = mpas_ds.sizes["nVertices"]
        assert len(node_lon) == n_vertices
        assert len(node_lat) == n_vertices

    def test_node_ranges(self, mpas_ds: xr.Dataset):
        """Node coordinates must be in valid degree ranges."""
        node_lon, node_lat, _, _, _, _ = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        assert np.all(node_lon >= 0) and np.all(node_lon <= 360)
        assert np.all(node_lat >= -90) and np.all(node_lat <= 90)

    def test_all_elements_are_triangles(self, mpas_ds: xr.Dataset):
        """Fan-triangulation should produce only TRI elements."""
        _, _, _, elem_types, _, _ = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        assert np.all(elem_types == esmpy.MeshElemType.TRI)

    def test_connectivity_indices_valid(self, mpas_ds: xr.Dataset):
        """All connectivity indices must reference valid nodes (0-based)."""
        node_lon, _, elem_conn, _, _, _ = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        assert elem_conn.min() >= 0
        assert elem_conn.max() < len(node_lon)

    def test_triangle_count_matches_formula(self, mpas_ds: xr.Dataset):
        """Number of triangles = sum(nEdgesOnCell - 2)."""
        _, _, _, _, elem_ids, orig_idx = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        n_edges = mpas_ds["nEdgesOnCell"].values
        expected_tris = int(np.sum(n_edges - 2))
        assert len(elem_ids) == expected_tris
        assert len(orig_idx) == expected_tris

    def test_orig_idx_covers_all_cells(self, mpas_ds: xr.Dataset):
        """Every original cell must appear in the triangulation mapping."""
        _, _, _, _, _, orig_idx = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        n_cells = mpas_ds.sizes["nCells"]
        assert orig_idx.min() >= 0
        assert orig_idx.max() < n_cells
        assert len(np.unique(orig_idx)) == n_cells

    def test_element_ids_unique_and_sequential(self, mpas_ds: xr.Dataset):
        """Element IDs must be 1-based and sequential."""
        _, _, _, _, elem_ids, _ = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        np.testing.assert_array_equal(
            elem_ids, np.arange(1, len(elem_ids) + 1, dtype=np.int32)
        )

    def test_connectivity_dtype(self, mpas_ds: xr.Dataset):
        """All arrays must be int32 as required by ESMF."""
        _, _, elem_conn, elem_types, elem_ids, orig_idx = (
            _get_unstructured_mesh_info(mpas_ds, method="conservative")
        )
        assert elem_conn.dtype == np.int32
        assert elem_types.dtype == np.int32
        assert elem_ids.dtype == np.int32
        assert orig_idx.dtype == np.int32


# ---------------------------------------------------------------------------
# 3. ESMF Mesh creation via _create_esmf_grid
# ---------------------------------------------------------------------------


class TestRealMPASEsmfMeshCreation:
    """Validate _create_esmf_grid produces an esmpy.Mesh for MPAS data.

    On Apple Silicon the mock is used; on CI (Linux) the real ESMF
    Mesh constructor is exercised end-to-end.
    """

    def test_returns_mesh_object(self, mpas_ds: xr.Dataset):
        """_create_esmf_grid must return an esmpy.Mesh (real or mock)."""
        grid_obj, provenance, orig_idx = _create_esmf_grid(
            mpas_ds, method="conservative", is_source=True
        )
        assert isinstance(grid_obj, esmpy.Mesh)
        assert orig_idx is not None

    def test_orig_idx_consistent(self, mpas_ds: xr.Dataset):
        """orig_idx from _create_esmf_grid must match _get_unstructured_mesh_info."""
        _, _, _, _, _, orig_idx_direct = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        _, _, orig_idx_grid = _create_esmf_grid(
            mpas_ds, method="conservative", is_source=True
        )
        np.testing.assert_array_equal(orig_idx_grid, orig_idx_direct)

    @pytest.mark.skipif(
        IS_APPLE_SILICON,
        reason="esmpy Mesh.add_elements crashes on Apple Silicon",
    )
    def test_real_esmf_node_count(self, mpas_ds: xr.Dataset):
        """ESMF Mesh node count must match the MPAS vertex count (real ESMF only)."""
        grid_obj, _, _ = _create_esmf_grid(
            mpas_ds, method="conservative", is_source=True
        )
        assert grid_obj.node_count == mpas_ds.sizes["nVertices"]

    @pytest.mark.skipif(
        IS_APPLE_SILICON,
        reason="esmpy Mesh.add_elements crashes on Apple Silicon",
    )
    def test_real_esmf_element_count(self, mpas_ds: xr.Dataset):
        """ESMF Mesh element count must match the triangulated count (real ESMF only)."""
        grid_obj, _, orig_idx = _create_esmf_grid(
            mpas_ds, method="conservative", is_source=True
        )
        n_edges = mpas_ds["nEdgesOnCell"].values
        expected_tris = int(np.sum(n_edges - 2))
        assert grid_obj.element_count == expected_tris
        assert len(orig_idx) == expected_tris


# ---------------------------------------------------------------------------
# 4. Cross-validation: polygon extraction vs ESMF arrays
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Ensure polygon extraction and ESMF Mesh arrays agree on topology."""

    def test_polygon_cells_match_esmf_orig_idx(self, mpas_ds: xr.Dataset):
        """Number of polygons must equal unique original cells in the
        ESMF triangulation."""
        polys = _extract_cell_polygons(mpas_ds)
        _, _, _, _, _, orig_idx = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        assert len(polys) == len(np.unique(orig_idx))

    def test_node_coords_consistent(self, mpas_ds: xr.Dataset):
        """Node coordinates used by polygon extraction and ESMF must match."""
        node_lon_esmf, node_lat_esmf, _, _, _, _ = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )

        # Recompute from raw MPAS the same way _extract_cell_polygons does
        node_lat_viz = _clip_latitudes(
            _to_degrees(mpas_ds["latVertex"])
        ).values
        node_lon_viz = _normalize_longitudes(
            _to_degrees(mpas_ds["lonVertex"])
        ).values

        np.testing.assert_allclose(node_lon_esmf, node_lon_viz, atol=1e-10)
        np.testing.assert_allclose(node_lat_esmf, node_lat_viz, atol=1e-10)

    def test_polygon_vertices_reference_same_nodes(self, mpas_ds: xr.Dataset):
        """Each polygon vertex must correspond to a node in the ESMF arrays."""
        node_lon, node_lat, _, _, _, _ = _get_unstructured_mesh_info(
            mpas_ds, method="conservative"
        )
        polys = _extract_cell_polygons(mpas_ds)

        # Build a set of (lon, lat) from ESMF nodes for fast lookup
        node_set = set(zip(np.round(node_lon, 8), np.round(node_lat, 8)))

        # Check a sample of cells (checking all 2562 is slow)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(polys), size=min(100, len(polys)), replace=False)
        for i in sample_idx:
            for lon_v, lat_v in polys[i]:
                # Dateline-shifted lons need to be wrapped back
                lon_check = lon_v % 360 if lon_v >= 360 else lon_v
                assert (round(lon_check, 8), round(lat_v, 8)) in node_set, (
                    f"Cell {i} vertex ({lon_v}, {lat_v}) not found in ESMF nodes"
                )


# ---------------------------------------------------------------------------
# 5. Visualization with the real mesh
# ---------------------------------------------------------------------------


class TestRealMPASPlotMesh:
    """Integration test: render the real MPAS mesh with plot_mesh."""

    def test_plot_mesh_orthographic(self, mpas_ds: xr.Dataset):
        """Default Orthographic projection should render without error."""
        coll = plot_mesh(mpas_ds, title="MPAS x1.2562 (Orthographic)")
        assert coll is not None
        plt.close("all")

    def test_plot_mesh_platecarree(self, mpas_ds: xr.Dataset):
        """PlateCarree projection should render without error."""
        coll = plot_mesh(
            mpas_ds,
            projection=ccrs.PlateCarree(),
            title="MPAS x1.2562 (PlateCarree)",
        )
        assert coll is not None
        plt.close("all")

    def test_plot_mesh_robinson(self, mpas_ds: xr.Dataset):
        """Robinson projection should render without error."""
        coll = plot_mesh(
            mpas_ds,
            projection=ccrs.Robinson(),
            edgecolor="steelblue",
            linewidth=0.2,
            title="MPAS x1.2562 (Robinson)",
        )
        assert coll is not None
        plt.close("all")

    def test_plot_mesh_saves_png(self, mpas_ds: xr.Dataset, tmp_path):
        """Verify we can save the mesh plot to a PNG file."""
        coll = plot_mesh(mpas_ds, title="MPAS x1.2562")
        out = tmp_path / "mpas_mesh.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        assert out.exists()
        assert out.stat().st_size > 0
