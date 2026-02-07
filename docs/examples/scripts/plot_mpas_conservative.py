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

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xregrid import Regridder, create_global_grid

# 1. Create a synthetic MPAS-like dataset
# We'll create a small regional mesh to avoid complexities with poles
# and prime meridian in this synthetic example.
n_lon, n_lat = 10, 8
# Stay away from poles and prime meridian wrap for a simple synthetic mesh
lon_edges = np.linspace(10, 170, n_lon + 1)
lat_edges = np.linspace(-60, 60, n_lat + 1)
lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
lon_v_mesh, lat_v_mesh = np.meshgrid(lon_edges, lat_edges)

nCells = n_lon * n_lat
nVertices = (n_lon + 1) * (n_lat + 1)

# Map (j, i) to vertex index (1-based for MPAS)
v_idx = np.arange(1, nVertices + 1).reshape(n_lat + 1, n_lon + 1)

verticesOnCell = np.zeros((nCells, 4), dtype=int)
for j in range(n_lat):
    for i in range(n_lon):
        idx = j * n_lon + i
        # Counter-clockwise: (j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)
        verticesOnCell[idx] = [
            v_idx[j, i],
            v_idx[j, i + 1],
            v_idx[j + 1, i + 1],
            v_idx[j + 1, i],
        ]

ds_mpas = xr.Dataset(
    {"data": (["nCells"], np.random.rand(nCells))},
    coords={
        "latCell": (["nCells"], np.radians(lat_mesh.flatten()), {"units": "radians"}),
        "lonCell": (["nCells"], np.radians(lon_mesh.flatten()), {"units": "rad"}),
        "latVertex": (
            ["nVertices"],
            np.radians(lat_v_mesh.flatten()),
            {"units": "radians"},
        ),
        "lonVertex": (
            ["nVertices"],
            np.radians(lon_v_mesh.flatten()),
            {"units": "rad"},
        ),
        "verticesOnCell": (["nCells", "maxNodes"], verticesOnCell),
        "nEdgesOnCell": (["nCells"], np.full(nCells, 4)),
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
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
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
