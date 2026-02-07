"""
Conservative MPAS Regridding
============================

This example demonstrates how to perform conservative regridding from an MPAS
unstructured grid to a standard rectilinear grid. Conservative regridding
is essential for flux variables like precipitation or heat flux to ensure
the total quantity is preserved.

XRegrid automatically handles:
- Coordinate conversion from radians to degrees
- Mesh construction from MPAS connectivity (verticesOnCell)
- Triangulation of MPAS polygons for ESMF compatibility
- Weight aggregation back to original cells
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder, create_global_grid

# 1. Create a synthetic MPAS-like dataset
# We define a small mesh with nodes and cells
nCells = 200
nVertices = 400

# Random cell centers
latCell = np.radians(np.random.uniform(-90, 90, nCells))
lonCell = np.radians(np.random.uniform(0, 360, nCells))

# Random vertices
latVertex = np.radians(np.random.uniform(-90, 90, nVertices))
lonVertex = np.radians(np.random.uniform(0, 360, nVertices))

# Mock connectivity (each cell has 6 vertices)
verticesOnCell = np.random.randint(1, nVertices + 1, (nCells, 6))

ds_mpas = xr.Dataset(
    {"data": (["nCells"], np.random.rand(nCells))},
    coords={
        "latCell": (["nCells"], latCell, {"units": "radians"}),
        "lonCell": (["nCells"], lonCell, {"units": "rad"}),
        "latVertex": (["nVertices"], latVertex, {"units": "radians"}),
        "lonVertex": (["nVertices"], lonVertex, {"units": "rad"}),
        "verticesOnCell": (["nCells", "maxNodes"], verticesOnCell),
        "nEdgesOnCell": (["nCells"], np.full(nCells, 6)),
    },
)

# 2. Define a rectilinear target grid (e.g., 5° global)
target_grid = create_global_grid(5, 5)

# 3. Create the regridder using the 'conservative' method
# XRegrid will detect the MPAS connectivity and use ESMF Mesh
regridder = Regridder(ds_mpas, target_grid, method="conservative")

# 4. Apply regridding
regridded = regridder(ds_mpas.data)

# 5. Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot unstructured centers
sc = ax1.scatter(
    np.degrees(ds_mpas.lonCell),
    np.degrees(ds_mpas.latCell),
    c=ds_mpas.data,
    s=50,
    cmap="Spectral_r",
)
ax1.set_title("MPAS Cell Centers (Source)")
plt.colorbar(sc, ax=ax1)

# Plot regridded result
regridded.plot(ax=ax2, cmap="Spectral_r")
ax2.set_title("Conservative Regridded (Target 5°)")

plt.tight_layout()
plt.show()

print("\nConservative Regridding Summary:")
print(f"Source: MPAS ({nCells} cells, {nVertices} vertices)")
print(f"Target: Rectilinear ({target_grid.sizes['lat']}x{target_grid.sizes['lon']})")
print(f"Method: {regridder.method}")
