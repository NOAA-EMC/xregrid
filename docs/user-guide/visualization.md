# Visualization Guide

XRegrid provides high-level visualization tools designed to help you verify and communicate your regridding results. We follow a **Two-Track Rule** for all plotting functions, ensuring they are suitable for both exploration and publication.

## The Two-Track Rule

Every plotting function in XRegrid supports two modes:

1.  **Track A: Static (`mode='static'`)**:
    - **Backends**: Matplotlib and Cartopy.
    - **Goal**: High-quality, publication-ready figures.
    - **Default**: This is the default mode for all functions.

2.  **Track B: Interactive (`mode='interactive'`)**:
    - **Backends**: HvPlot and HoloViews.
    - **Goal**: Exploratory data analysis, zooming, and point inspection.
    - **Requirement**: Requires `hvplot` and `holoviews` installed.

---

## 1. Regridding Comparison

The `plot_comparison` method provides a side-by-side view of the source grid, the target grid, and the difference.

### Static Comparison

```python
import xarray as xr
from xregrid import Regridder

import numpy as np

# Setup regridder and data
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
target = xr.Dataset({"lat": (["lat"], np.arange(15, 76, 1.0)), "lon": (["lon"], np.arange(200, 331, 1.0))})
regridder = Regridder(ds, target)
regridded = regridder(ds.air)

# Plot static comparison (publication-quality)
regridder.plot_comparison(ds.air, regridded, mode='static')
```

### Interactive Comparison

```python
# Plot interactive comparison (exploratory)
regridder.plot_comparison(ds.air, regridded, mode='interactive')
```

---

## 2. Weight Diagnostics

Visualizing your regridding weights is essential for scientific hygiene. The `plot_diagnostics` method helps you identify "unmapped" regions and areas with poor weight sums.

```python
# Static diagnostics
regridder.plot_diagnostics(mode='static')

# Interactive diagnostics
regridder.plot_diagnostics(mode='interactive')
```

- **Weight Sum**: Should ideally be 1.0 everywhere. Values far from 1.0 indicate interpolation issues or boundary effects.
- **Unmapped Mask**: Identifies destination cells that do not overlap with any source cells.

---

## 3. Weight Inspection

To understand how specific destination points are interpolated, you can visualize the source points that contribute to them using `plot_weights`.

```python
# Visualize weights contributing to destination point index 500
regridder.plot_weights(row_idx=500, mode='static')
```

---

## 4. General Spatial Plotting

XRegrid also provides a standalone `plot` function that follows the Two-Track Rule for any xarray DataArray. It automatically handles CRS discovery and spatial slicing of multi-dimensional data.

```python
from xregrid import plot

# Automatically detects CRS and slices N-D data to 2D for plotting
plot(regridded, mode='static', cmap='viridis')

# Interactive version with automated rasterization for large datasets
plot(regridded, mode='interactive')
```

---

## Customizing Plots

Since Track A uses standard Matplotlib and Track B uses standard HvPlot, you can pass additional keyword arguments to customize the results:

### Track A (Static) Customization

```python
import cartopy.crs as ccrs

regridder.plot_comparison(
    ds.air, regridded,
    projection=ccrs.Orthographic(-100, 45), # Change map projection
    cmap='coolwarm',                         # Change colormap
    figsize=(15, 4)                          # Figure size
)
```

### Track B (Interactive) Customization

```python
regridder.plot_comparison(
    ds.air, regridded,
    mode='interactive',
    rasterize=False, # Disable rasterization for small grids
    width=400,       # Set individual plot width
    height=300
)
```

## 5. Mesh Wireframe Visualization

For unstructured grids (MPAS, UGRID, SCRIP), `plot_mesh` draws the cell edges on a map projection — useful for inspecting variable-resolution meshes before regridding.

```python
import xarray as xr
from xregrid.viz import plot_mesh

ds = xr.open_dataset("mpas_mesh.nc")

# Default: Orthographic globe view
plot_mesh(ds, title="MPAS Variable-Resolution Mesh")
```

You can customize the projection, edge styling, and even fill cells:

```python
import cartopy.crs as ccrs

# Flat map with colored edges
plot_mesh(
    ds,
    projection=ccrs.Robinson(),
    edgecolor="steelblue",
    linewidth=0.2,
    title="MPAS Mesh (Robinson)",
)

# Filled cells for density visualization
plot_mesh(
    ds,
    facecolor="lightskyblue",
    edgecolor="navy",
    alpha=0.4,
)
```

`plot_mesh` supports any dataset that contains MPAS (`verticesOnCell`), UGRID (`face_node_connectivity`), or SCRIP (`lat_b`/`lon_b`) conventions.

---

## Summary of Plotting Functions

| Method | Description |
|--------|-------------|
| `Regridder.plot_comparison()` | Side-by-side: Source, Target, Difference |
| `Regridder.plot_diagnostics()`| Spatial quality (Weight Sum, Unmapped Mask) |
| `Regridder.plot_weights()`    | Inspect weights for a specific point |
| `xregrid.plot()`              | Generic 2D spatial plot for any DataArray |
| `xregrid.plot_mesh()`         | Unstructured mesh wireframe (MPAS, UGRID, SCRIP) |
