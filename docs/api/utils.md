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

### create_grid_from_crs

::: xregrid.create_grid_from_crs

Create a structured grid dataset from a Coordinate Reference System (CRS) and extent.

```python
from xregrid import create_grid_from_crs

# Create a Lambert Conformal Conic grid over North America
extent = (-2500000, 2500000, -2000000, 2000000)
res = (12000, 12000) # 12km
crs = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

ds = create_grid_from_crs(crs, extent, res)
```

### create_grid_from_ioapi

::: xregrid.create_grid_from_ioapi

Create a structured grid dataset from IOAPI-compliant metadata.

```python
from xregrid.utils import create_grid_from_ioapi

metadata = {
    "GDTYP": 2,
    "P_ALP": 30.0,
    "P_BET": 60.0,
    "XCENT": -97.0,
    "YCENT": 40.0,
    "XORIG": -1000.0,
    "YORIG": -1000.0,
    "XCELL": 500.0,
    "YCELL": 500.0,
    "NCOLS": 100,
    "NROWS": 100,
}

ds = create_grid_from_ioapi(metadata)
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
