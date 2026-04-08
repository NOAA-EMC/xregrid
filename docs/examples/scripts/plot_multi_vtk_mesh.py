"""
Multi-VTK Mesh Overlay
======================

This example demonstrates how to plot multiple VTK mesh files on a
single map. This is useful for comparing source and target meshes,
visualizing nested grids, or inspecting multi-resolution domains.

Key concepts:
- Loading VTK legacy files with ``load_vtk_mesh``
- Plotting multiple meshes on a shared axes with ``plot_mesh``
- Using color and linewidth to distinguish overlaid meshes
"""

import os
import tempfile

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from xregrid.utils import load_vtk_mesh
from xregrid.viz import plot_mesh

# %%
# 1. Generate synthetic VTK mesh files
# --------------------------------------
# We create three meshes at different resolutions to simulate a
# coarse global mesh, a regional refinement, and a fine local nest.


def write_vtk_hex_mesh(path, center_lon, center_lat, radius, n_rings, label="mesh"):
    """
    Write a simple hexagonal-ish VTK mesh around a centre point.

    Parameters
    ----------
    path : str
        Output .vtk file path.
    center_lon, center_lat : float
        Centre of the mesh in degrees.
    radius : float
        Approximate radius in degrees.
    n_rings : int
        Number of concentric rings of cells.
    label : str
        Description string for the VTK header.
    """
    cells = []
    points = []

    for ring in range(1, n_rings + 1):
        n_cells_in_ring = 6 * ring
        for k in range(n_cells_in_ring):
            angle = 2 * np.pi * k / n_cells_in_ring
            r_inner = radius * (ring - 1) / n_rings
            r_outer = radius * ring / n_rings
            da = np.pi / n_cells_in_ring

            # Four corners of a trapezoidal cell
            p0 = (
                center_lon + r_inner * np.cos(angle - da),
                center_lat + r_inner * np.sin(angle - da),
            )
            p1 = (
                center_lon + r_inner * np.cos(angle + da),
                center_lat + r_inner * np.sin(angle + da),
            )
            p2 = (
                center_lon + r_outer * np.cos(angle + da),
                center_lat + r_outer * np.sin(angle + da),
            )
            p3 = (
                center_lon + r_outer * np.cos(angle - da),
                center_lat + r_outer * np.sin(angle - da),
            )

            base = len(points)
            points.extend([p0, p1, p2, p3])
            cells.append([base, base + 1, base + 2, base + 3])

    n_points = len(points)
    n_cells = len(cells)

    lines = [
        "# vtk DataFile Version 3.0",
        label,
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} double",
    ]
    for lon, lat in points:
        lines.append(f"{lon:.8f} {lat:.8f} 0.0")

    total_ints = n_cells * 5  # 4 verts + count per cell
    lines.append(f"CELLS {n_cells} {total_ints}")
    for cell in cells:
        lines.append(f"4 {cell[0]} {cell[1]} {cell[2]} {cell[3]}")

    lines.append(f"CELL_TYPES {n_cells}")
    for _ in range(n_cells):
        lines.append("9")  # VTK_QUAD

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


tmpdir = tempfile.mkdtemp()

# Coarse global-ish mesh centred on the Atlantic
vtk_coarse = os.path.join(tmpdir, "coarse.vtk")
write_vtk_hex_mesh(vtk_coarse, -30, 20, radius=60, n_rings=4, label="Coarse mesh")

# Medium regional mesh over Europe
vtk_medium = os.path.join(tmpdir, "medium.vtk")
write_vtk_hex_mesh(vtk_medium, 10, 48, radius=20, n_rings=5, label="Medium mesh")

# Fine local mesh over the Alps
vtk_fine = os.path.join(tmpdir, "fine.vtk")
write_vtk_hex_mesh(vtk_fine, 10, 46, radius=5, n_rings=6, label="Fine mesh")

# %%
# 2. Load and inspect
# --------------------
# ``load_vtk_mesh`` converts each VTK file into a UGRID-style Dataset.

ds_coarse = load_vtk_mesh(vtk_coarse)
ds_medium = load_vtk_mesh(vtk_medium)
ds_fine = load_vtk_mesh(vtk_fine)

print(f"Coarse: {ds_coarse.sizes['nFaces']} cells, {ds_coarse.sizes['nNodes']} nodes")
print(f"Medium: {ds_medium.sizes['nFaces']} cells, {ds_medium.sizes['nNodes']} nodes")
print(f"Fine:   {ds_fine.sizes['nFaces']} cells, {ds_fine.sizes['nNodes']} nodes")

# %%
# 3. Overlay on a single map
# ----------------------------
# Pass a shared ``ax`` to ``plot_mesh`` to layer multiple meshes.
# Use distinct colors and linewidths to tell them apart.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Coarse mesh — light gray, thin
plot_mesh(ds_coarse, ax=ax, edgecolor="silver", linewidth=0.4, facecolor="none")

# Medium mesh — blue
plot_mesh(ds_medium, ax=ax, edgecolor="steelblue", linewidth=0.6, facecolor="none")

# Fine mesh — red, thicker
plot_mesh(ds_fine, ax=ax, edgecolor="firebrick", linewidth=0.8, facecolor="none")

ax.set_extent([-100, 60, -30, 80], crs=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
ax.set_title("Multi-Resolution Mesh Overlay (3 VTK files)")

# Legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="silver", lw=1.5, label="Coarse"),
    Line2D([0], [0], color="steelblue", lw=1.5, label="Medium"),
    Line2D([0], [0], color="firebrick", lw=1.5, label="Fine"),
]
ax.legend(handles=legend_elements, loc="lower left")

plt.tight_layout()
plt.show()

# %%
# 4. Side-by-side comparison
# ----------------------------
# Alternatively, plot each mesh in its own panel.

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

meshes = [
    (ds_coarse, "Coarse (VTK)", "gray"),
    (ds_medium, "Medium (VTK)", "steelblue"),
    (ds_fine, "Fine (VTK)", "firebrick"),
]

for ax, (ds, title, color) in zip(axes, meshes):
    plot_mesh(ds, ax=ax, edgecolor=color, linewidth=0.5, title=title)
    ax.coastlines(linewidth=0.3)
    ax.set_extent([-100, 60, -30, 80], crs=ccrs.PlateCarree())

plt.tight_layout()
plt.show()

# %%
# 5. Direct from VTK file paths
# --------------------------------
# ``plot_mesh`` also accepts a file path string directly, so you
# can skip the ``load_vtk_mesh`` step for quick inspection.

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(10, 46))

for vtk_path, color, lw in [
    (vtk_coarse, "silver", 0.3),
    (vtk_medium, "steelblue", 0.5),
    (vtk_fine, "firebrick", 0.7),
]:
    plot_mesh(vtk_path, ax=ax, edgecolor=color, linewidth=lw)

ax.set_title("Direct from VTK file paths (Orthographic)")
plt.tight_layout()
plt.show()
