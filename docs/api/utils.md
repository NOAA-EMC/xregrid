# Utilities

XRegrid provides several utility functions for creating standard grids and loading ESMF-formatted files.

## Grid Generation

### create_global_grid

::: xregrid.create_global_grid

Create a global rectilinear grid dataset with a specified resolution.

```python
from xregrid import create_global_grid

# Create a 1x1 degree global grid with bounds
ds = create_global_grid(res_lat=1.0, res_lon=1.0)
```

### create_regional_grid

::: xregrid.create_regional_grid

Create a regional rectilinear grid dataset for a specific geographic bounding box.

```python
from xregrid import create_regional_grid

# Create a regional grid over Europe
ds = create_regional_grid(
    lat_range=(35, 70),
    lon_range=(-10, 40),
    res_lat=0.25,
    res_lon=0.25
)
```

## ESMF File Support

### load_esmf_file

::: xregrid.load_esmf_file

Load an ESMF mesh, mosaic, or grid file into an xarray Dataset.

```python
from xregrid import load_esmf_file

# Load an ESMF mesh file
ds = load_esmf_file("path/to/mesh.nc")
```
