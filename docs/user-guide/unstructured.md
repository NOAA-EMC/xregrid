# Unstructured Grids

XRegrid provides robust support for unstructured grids, including those used by models like MPAS and ICON, or datasets following UGRID conventions. It also provides native integration with **uxarray**.

## Supported Formats

XRegrid automatically detects unstructured grids by looking for:
- Standard 1D coordinates sharing a dimension (e.g., `lat(nCells)`, `lon(nCells)`).
- **MPAS-specific** coordinate names: `latCell`, `lonCell`, `latVertex`, `lonVertex`.
- **UGRID-specific** coordinate names: `lat_node`, `lon_node`.
- **UGRID Topology**: Robustly identifies `mesh_topology` variables with `cf_role="mesh_topology"` and associated connectivity roles.
- **uxarray** objects: Automatically uses the underlying `.uxgrid` for coordinates and connectivity.

## Regridding Methods

### Nearest Neighbor (`nearest_s2d`, `nearest_d2s`)
For simple point-to-grid or grid-to-point regridding where connectivity is not available, XRegrid uses ESMF's `LocStream` interface.

### Conservative Regridding (`conservative`)
XRegrid now supports **conservative regridding** for unstructured grids by automatically constructing an ESMF `Mesh`. This requires connectivity information in the dataset:

- **MPAS**: Requires `verticesOnCell` and vertex coordinates (`latVertex`, `lonVertex`).
- **UGRID**: Requires `face_node_connectivity` identified either by name or via the `cf_role="face_node_connectivity"` attribute.
- **uxarray**: Automatically extracts connectivity from the `UxDataset` or `UxDataArray`.

XRegrid automatically triangulates these polygons to ensure compatibility with ESMF's mesh requirements while correctly aggregating weights back to the original cells.

## Scientific Hygiene

XRegrid ensures that UGRID-specific metadata is preserved during regridding. If the target is also an unstructured grid following UGRID conventions, the output dataset will include the appropriate `mesh` and `location` attributes, as well as the `mesh_topology` variable.

## Dask Parallel Support

Weight generation for unstructured grids can be fully parallelized across a Dask cluster by setting `parallel=True` in the `Regridder`.

```python
import xarray as xr
from xregrid import Regridder

# Load MPAS or UGRID dataset
ds = xr.open_dataset("mpas_data.nc")
target_grid = xr.open_dataset("target_grid.nc")

# Create regridder with Dask parallel weight generation
regridder = Regridder(ds, target_grid, method='conservative', parallel=True)

# Apply regridding
regridded_data = regridder(ds.some_variable)
```

## Coordinate Units

XRegrid automatically detects if coordinates are in radians (using the `units` attribute) and converts them to degrees before passing them to ESMF. This is common in MPAS datasets.

## Performance Tips

- For very large unstructured meshes (millions of cells), always use `parallel=True` if a Dask cluster is available.
- If you plan to regrid multiple variables or time steps on the same mesh, use the `reuse_weights=True` and `filename` parameters to cache the calculated weights.
