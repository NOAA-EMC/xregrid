"""
Unstructured Mesh Wireframe
============================

This example shows how to visualize the cell structure of an unstructured
mesh using ``plot_mesh``. This is useful for inspecting variable-resolution
grids like MPAS, UGRID, or SCRIP meshes before regridding.

The wireframe view draws each cell polygon on a map projection, similar
to the classic MPAS variable-resolution mesh images from NCAR.

Key concepts:
- Visualizing mesh topology without data
- Comparing projections (Orthographic vs PlateCarree)
- Customizing edge color, linewidth, and fill
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xregrid.viz import plot_mesh

# %%
# 1. Build a synthetic MPAS-style mesh
# -------------------------------------
# A small icosahedral-like mesh with hexagons and pentagons.

np.random.seed(0)
n_cells = 200
n_vertices_per_cell = 6
n_vertices = n_cells * n_vertices_per_cell

# Random cell centres spread across the globe
cell_lat = np.random.uniform(-80, 80, n_cells)
cell_lon = np.random.uniform(0, 360, n_cells)

# Build vertices around each centre
vert_lat = np.zeros(n_vertices)
vert_lon = np.zeros(n_vertices)
conn = np.zeros((n_cells, n_vertices_per_cell), dtype=int)
n_edges = np.full(n_cells, n_vertices_per_cell, dtype=int)

for i in range(n_cells):
    angles = np.linspace(0, 2 * np.pi, n_vertices_per_cell, endpoint=False)
    radius = 3.0
    base = i * n_vertices_per_cell
    vert_lat[base : base + n_vertices_per_cell] = cell_lat[i] + radius * np.sin(
        angles
    )
    vert_lon[base : base + n_vertices_per_cell] = cell_lon[i] + radius * np.cos(
        angles
    )
    conn[i] = np.arange(base, base + n_vertices_per_cell) + 1  # 1-based

ds_mesh = xr.Dataset(
    {
        "latVertex": (["nVertices"], np.radians(vert_lat)),
        "lonVertex": (["nVertices"], np.radians(vert_lon)),
        "verticesOnCell": (["nCells", "maxEdges"], conn),
        "nEdgesOnCell": (["nCells"], n_edges),
    }
)

# %%
# 2. Orthographic projection (globe view)
# -----------------------------------------
# The default projection gives a 3D globe perspective.

plot_mesh(ds_mesh, title="Synthetic MPAS Mesh (Orthographic)")
plt.show()

# %%
# 3. PlateCarree projection (flat map)
# --------------------------------------
# A flat equirectangular view is useful for inspecting cell sizes
# across latitudes.

plot_mesh(
    ds_mesh,
    projection=ccrs.PlateCarree(),
    edgecolor="steelblue",
    linewidth=0.5,
    title="Synthetic MPAS Mesh (PlateCarree)",
)
plt.show()

# %%
# 4. Filled cells with transparency
# -----------------------------------
# Setting ``facecolor`` fills the cells, which can help highlight
# mesh density variations.

plot_mesh(
    ds_mesh,
    projection=ccrs.Robinson(),
    facecolor="lightskyblue",
    edgecolor="navy",
    alpha=0.4,
    linewidth=0.3,
    title="Filled Mesh Cells (Robinson)",
)
plt.show()
