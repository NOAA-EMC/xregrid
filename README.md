# XRegrid

An optimized ESMF-based regridder for xarray that provides significant performance improvements over xESMF.

## Overview

XRegrid is a high-performance regridding library that builds on top of ESMF (Earth System Modeling Framework) to provide fast and accurate interpolation between different grids. It offers substantial performance improvements over existing solutions while maintaining full compatibility with xarray data structures.

## Key Features

- **High Performance**: Up to 35x faster than xESMF for single time-step regridding
- **Correct ESMF Integration**: Native support for rectilinear and curvilinear grids
- **Dask Integration**: Seamless parallel processing with Dask arrays
- **Memory Efficient**: Optimized sparse matrix operations using scipy
- **xarray Compatible**: Native support for xarray datasets and data arrays
- **Automatic coordinate detection**: Support for `cf-xarray` for easy coordinate and boundary identification
- **Weight Reuse**: Save and load regridding weights to/from NetCDF files
- **Grid Utilities**: Built-in functions for quick global and regional grid generation

## Quick Example

```python
import xarray as xr
from xregrid import Regridder

# Load tutorial data
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)

# Define a target grid (e.g., 1.0° resolution)
import numpy as np
target_grid = xr.Dataset({
    "lat": (["lat"], np.arange(15, 76, 1.0)),
    "lon": (["lon"], np.arange(200, 331, 1.0))
})

# Create regridder and apply
regridder = Regridder(ds, target_grid, method='bilinear')
air_regridded = regridder(ds.air)
```

## Installation

Install via mamba (recommended):

```bash
mamba env create -f environment.yml
mamba activate xregrid
```

Or install from source:

```bash
pip install .
```

## Documentation

Full documentation is available at [https://xregrid.readthedocs.io](https://xregrid.readthedocs.io)

- [Quick Start Guide](https://xregrid.readthedocs.io/user-guide/quickstart/)
- [API Reference](https://xregrid.readthedocs.io/api/regridder/)

## Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues or pull requests.

## License

XRegrid is released under the MIT License.
