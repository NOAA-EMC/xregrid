# Command Line Interface (CLI)

`xregrid` provides a powerful command-line interface for regridding NetCDF files without writing any Python code.

## Basic Usage

The basic syntax for the CLI is:

```bash
xregrid <src_file> <target> [options]
```

Where:
- `<src_file>` is the path to your source NetCDF file.
- `<target>` can be either:
    - A path to a target NetCDF file defining the grid.
    - A numerical resolution in degrees (e.g., `1.0` for a 1-degree grid).

### Examples

**Regrid to a 1-degree global grid:**

```bash
xregrid input.nc 1.0 -o output.nc
```

**Regrid to a specific target file:**

```bash
xregrid source.nc target_grid.nc -o output.nc
```

**Regrid to a 0.5-degree regional grid:**

```bash
xregrid input.nc 0.5 --extent=-10,10,20,40 -o regional_output.nc
```

## Options

### General Options

- `--method`: Regridding method. Choices: `bilinear`, `conservative`, `nearest_s2d`, `nearest_d2s`, `patch` (default: `bilinear`).
- `--output`, `-o`: Path to the output NetCDF file (default: `output.nc`).
- `--extent`: Target grid extent as `min_lat,max_lat,min_lon,max_lon`. Only used if `<target>` is a resolution.
- `--periodic`: Set for global grids with periodic boundaries.
- `--skipna`: Handle NaNs by re-normalizing weights.
- `--reuse-weights`: Reuse weights if the weights file already exists.
- `--weights-file`: Path to the weights file (default: `weights.nc`).

### Dask Parallelization

The CLI can leverage Dask to parallelize weight generation and application, which is essential for very large datasets.

- `--dask-local <N>`: Start a local Dask cluster with `N` workers.
- `--dask-scheduler <ADDRESS>`: Connect to an existing Dask scheduler.
- `--dask-jobqueue <MACHINE>`: Use `dask-jobqueue` to submit jobs to an HPC cluster (e.g., `hera`, `jet`).
- `--dask-account <ACCOUNT>`: Specify the SLURM account for `dask-jobqueue`.

#### Example with Local Dask Cluster

```bash
xregrid huge_file.nc 0.1 --dask-local 8 -o output.nc
```

## Tips

- When using `--extent` with negative values, use the equals sign to avoid parsing issues: `--extent=-90,90,0,360`.
- For global regridding, always use `--periodic` and ensure your target grid is global (default if only resolution is provided).
