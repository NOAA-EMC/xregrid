# XRegrid

An optimized ESMF-based regridder for xarray that provides significant performance improvements over xESMF.

## Overview

XRegrid is a high-performance regridding library that builds on top of ESMF (Earth System Modeling Framework) to provide fast and accurate interpolation between different grids. It offers substantial performance improvements over existing solutions while maintaining full compatibility with xarray data structures.

## Key Features

- **High Performance**: Up to 30x faster than xESMF for single time-step regridding
- **Correct ESMF Integration**: Native support for rectilinear and curvilinear grids
- **Dask Integration**: Seamless parallel processing with Dask arrays
- **Memory Efficient**: Optimized sparse matrix operations using scipy
- **xarray Compatible**: Native support for xarray datasets and data arrays
- **Weight Reuse**: Save and load regridding weights to/from NetCDF files

## Quick Example

```python
import xarray as xr
import numpy as np
from xregrid import ESMPyRegridder

# Create source and target grids
source_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 180)),
    'lon': (['lon'], np.linspace(0, 359, 360))
})

target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 360)),
    'lon': (['lon'], np.linspace(0, 359.5, 720))
})

# Create regridder
regridder = ESMPyRegridder(
    source_grid, target_grid,
    method='bilinear',
    periodic=True
)

# Apply to your data
temperature = xr.DataArray(
    np.random.rand(10, 180, 360),
    dims=['time', 'lat', 'lon'],
    coords={'lat': source_grid.lat, 'lon': source_grid.lon}
)

temperature_regridded = regridder(temperature)
```

## Performance Highlights

| Resolution | Grid Points | ESMPyRegridder | xESMF | Speedup |
|------------|-------------|----------------|--------|---------|
| 1.0° Global | 64,800 | 0.0027s | 0.044s | ~16x |
| 0.5° Global | 259,200 | 0.0073s | 0.178s | ~24x |
| 0.25° Global | 1,036,800 | 0.025s | 0.69s | ~27x |

## Installation

Install via conda (recommended):

```bash
conda env create -f environment.yml
conda activate xregrid
```

Or install from source:

```bash
pip install .
```

See the [Installation Guide](installation.md) for detailed instructions.

## Getting Started

- [Quick Start Guide](user-guide/quickstart.md) - Learn the basics
- [Examples Gallery](examples/index.md) - See XRegrid in action
- [Performance Guide](user-guide/performance.md) - Optimization tips
- [API Reference](api/regridder.md) - Complete API documentation

## Why XRegrid?

XRegrid was developed to address performance bottlenecks in existing regridding solutions for high-resolution climate data. By leveraging optimized sparse matrix operations and proper ESMF integration, XRegrid provides:

1. **Vectorized Operations**: Single large sparse matrix multiplications instead of loop-based approaches
2. **Optimized Memory Usage**: scipy sparse matrices with lower memory footprint
3. **Correct ESMF Integration**: Proper coordinate transposition and index alignment
4. **Dask Scalability**: Linear scaling with the number of chunks

## Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues or pull requests.

## License

XRegrid is released under the MIT License.