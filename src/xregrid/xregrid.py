from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix

try:
    import esmpy
except ImportError:
    esmpy = None

try:
    import dask
    from distributed import get_client
except ImportError:
    dask = None

from .utils import update_history

# Global cache for workers to reuse ESMF objects across tasks
_WORKER_CACHE: Dict[int, Any] = {}

if TYPE_CHECKING:
    pass


def _get_mesh_info(
    ds: xr.Dataset,
) -> Tuple[xr.DataArray, xr.DataArray, Tuple[int, ...], Tuple[str, ...], bool]:
    """
    Detect grid type and extract coordinates and shape.

    Uses cf-xarray for automatic coordinate detection if standard
    names 'lat' and 'lon' are not present.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to extract mesh info from.

    Returns
    -------
    lon : xarray.DataArray
        Longitude coordinate.
    lat : xarray.DataArray
        Latitude coordinate.
    shape : tuple of int
        Grid shape.
    dims : tuple of str
        Coordinate dimensions.
    is_unstructured : bool
        Whether the grid is unstructured.
    """
    try:
        lat = ds.cf["latitude"]
        lon = ds.cf["longitude"]
    except (KeyError, AttributeError):
        if "lat" in ds and "lon" in ds:
            lat = ds["lat"]
            lon = ds["lon"]
        else:
            raise KeyError(
                "Could not find latitude/longitude coordinates. "
                "Ensure they are named 'lat'/'lon' or have CF attributes."
            )

    if lat.ndim == 2:
        # Curvilinear
        if lon.ndim == 2 and lon.dims != lat.dims and set(lon.dims) == set(lat.dims):
            lon = lon.transpose(*lat.dims)
        return lon, lat, lat.shape, lat.dims, False
    elif lat.ndim == 1:
        if lat.dims == lon.dims:
            # Unstructured (e.g. MPAS)
            return lon, lat, lat.shape, lat.dims, True
        else:
            # Rectilinear
            lon_mesh, lat_mesh = xr.broadcast(lon, lat)

            # Ensure they have the correct order (lat, lon) for the shape
            if lat.ndim == 2 and lon.ndim == 2:
                if lat.dims != lon.dims and set(lat.dims) == set(lon.dims):
                    lon = lon.transpose(*lat.dims)

            lon_mesh = lon_mesh.transpose(lat.dims[0], lon.dims[0])
            lat_mesh = lat_mesh.transpose(lat.dims[0], lon.dims[0])

            return (
                lon_mesh,
                lat_mesh,
                (lat.size, lon.size),
                (lat.dims[0], lon.dims[0]),
                False,
            )
    else:
        raise ValueError("Latitude and longitude must be 1D or 2D.")


def _bounds_to_vertices(b: xr.DataArray) -> np.ndarray:
    """
    Convert bounds to vertices for ESMF.

    Handles 1D coordinates (N, 2) -> (N+1,) and 2D coordinates (Y, X, 4) -> (Y+1, X+1).

    Parameters
    ----------
    b : xarray.DataArray
        The bounds data array.

    Returns
    -------
    np.ndarray
        The vertex array.
    """
    if b.ndim == 2 and b.shape[-1] == 2:
        # 1D coordinates: (N, 2) -> (N+1,)
        return np.concatenate([b.values[:, 0], b.values[-1:, 1]])
    elif b.ndim == 3 and b.shape[-1] == 4:
        # 2D coordinates: (Y, X, 4) -> (Y+1, X+1)
        y_size, x_size, _ = b.shape
        vals = b.values
        res = np.empty((y_size + 1, x_size + 1))
        res[:-1, :-1] = vals[:, :, 0]
        res[:-1, -1] = vals[:, -1, 1]
        res[-1, -1] = vals[-1, -1, 2]
        res[-1, :-1] = vals[-1, :, 3]
        return res
    return b.values


def _get_grid_bounds(
    ds: xr.Dataset,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract grid cell boundaries from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract bounds from.

    Returns
    -------
    lat_b : np.ndarray or None
        Latitude boundaries.
    lon_b : np.ndarray or None
        Longitude boundaries.
    """
    try:
        lat_b_da = ds.cf.get_bounds("latitude")
        lon_b_da = ds.cf.get_bounds("longitude")
        return _bounds_to_vertices(lat_b_da), _bounds_to_vertices(lon_b_da)
    except (KeyError, AttributeError, ValueError):
        if "lat_b" in ds and "lon_b" in ds:
            lat_b = (
                ds["lat_b"].values if hasattr(ds["lat_b"], "values") else ds["lat_b"]
            )
            lon_b = (
                ds["lon_b"].values if hasattr(ds["lon_b"], "values") else ds["lon_b"]
            )
            return lat_b, lon_b
    return None, None


def _compute_chunk_weights(
    src_ds: xr.Dataset,
    tgt_chunk_ds: xr.Dataset,
    tgt_slice: dict,
    total_tgt_shape: tuple,
    method: str,
    periodic: bool,
    mask_var: Optional[str],
    extrap_method: Optional[str],
    extrap_dist_exponent: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker function to compute weights for a single target chunk.

    Parameters
    ----------
    src_ds : xr.Dataset
        Source grid dataset.
    tgt_chunk_ds : xr.Dataset
        Target grid chunk dataset.
    tgt_slice : dict
        Slices defining the target chunk in the global grid.
    total_tgt_shape : tuple
        Total shape of the target grid.
    method : str
        Regridding method.
    periodic : bool
        Whether the grid is periodic.
    mask_var : str, optional
        Name of the mask variable in source grid.
    extrap_method : str, optional
        Extrapolation method.
    extrap_dist_exponent : float
        Exponent for IDW extrapolation.

    Returns
    -------
    rows : np.ndarray
        Global destination indices.
    cols : np.ndarray
        Global source indices.
    weights : np.ndarray
        Regridding weights.
    """
    # Global cache for source ESMF object to avoid re-creation
    src_id = id(src_ds)
    if src_id in _WORKER_CACHE:
        src_obj = _WORKER_CACHE[src_id]
    else:
        src_obj = _create_esmf_grid(
            src_ds, method=method, periodic=periodic, mask_var=mask_var, is_source=True
        )
        _WORKER_CACHE[src_id] = src_obj

    # Create target chunk ESMF object
    tgt_obj = _create_esmf_grid(
        tgt_chunk_ds, method=method, periodic=False, is_source=False
    )

    src_field = esmpy.Field(src_obj, name="src")
    tgt_field = esmpy.Field(tgt_obj, name="tgt")

    regrid_method_map = {
        "bilinear": esmpy.RegridMethod.BILINEAR,
        "conservative": esmpy.RegridMethod.CONSERVE,
        "nearest_s2d": esmpy.RegridMethod.NEAREST_STOD,
        "nearest_d2s": esmpy.RegridMethod.NEAREST_DTOS,
        "patch": esmpy.RegridMethod.PATCH,
    }

    extrap_method_map = {
        "nearest_s2d": esmpy.ExtrapMethod.NEAREST_STOD,
        "nearest_idw": esmpy.ExtrapMethod.NEAREST_IDAVG,
        "creep_fill": esmpy.ExtrapMethod.CREEP_FILL,
    }

    regrid_kwargs = {
        "regrid_method": regrid_method_map[method],
        "unmapped_action": esmpy.UnmappedAction.IGNORE,
        "factors": True,
    }

    if extrap_method and extrap_method != "none":
        regrid_kwargs["extrap_method"] = extrap_method_map[extrap_method]
        regrid_kwargs["extrap_dist_exponent"] = extrap_dist_exponent

    if mask_var and mask_var in src_ds:
        regrid_kwargs["src_mask_values"] = np.array([0], dtype=np.int32)

    regrid = esmpy.Regrid(src_field, tgt_field, **regrid_kwargs)
    weights = regrid.get_weights_dict(deep_copy=True)

    # Convert local 1-based indices to global 0-based indices
    lat_dim = tgt_chunk_ds.cf["latitude"].dims[0]
    lon_dim = tgt_chunk_ds.cf["longitude"].dims[0]
    y_slice = tgt_slice[lat_dim]
    x_slice = tgt_slice[lon_dim]

    # Generate a mapping from local index to global flat index for the target
    y_indices, x_indices = np.meshgrid(
        np.arange(y_slice.start, y_slice.stop),
        np.arange(x_slice.start, x_slice.stop),
        indexing="ij",
    )
    global_tgt_indices = (y_indices * total_tgt_shape[1] + x_indices).flatten()

    # ESMF uses 1-based indexing
    rows = global_tgt_indices[weights["row_dst"] - 1]
    cols = weights["col_src"] - 1
    data = weights["weights"]

    return rows, cols, data


def _create_esmf_grid(
    ds: xr.Dataset,
    method: str = "bilinear",
    periodic: bool = False,
    mask_var: Optional[str] = None,
    is_source: bool = True,
) -> Union[esmpy.Grid, esmpy.LocStream]:
    """
    Creates an ESMF Grid or LocStream (Private helper).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to create ESMF object from.
    method : str
        Regridding method.
    periodic : bool
        Whether grid is periodic.
    mask_var : str, optional
        Mask variable name.
    is_source : bool
        Whether this is the source grid.

    Returns
    -------
    Union[esmpy.Grid, esmpy.LocStream]
        The ESMF object.
    """
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)

    if is_unstructured:
        if method not in ["nearest_s2d", "nearest_d2s"]:
            raise NotImplementedError(
                f"Method '{method}' is not yet supported for unstructured grids."
            )

        locstream = esmpy.LocStream(shape[0], coord_sys=esmpy.CoordSys.SPH_DEG)
        locstream["ESMF:Lon"] = lon.values.astype(np.float64)
        locstream["ESMF:Lat"] = lat.values.astype(np.float64)
        return locstream
    else:
        lon_f = lon.values.T
        lat_f = lat.values.T
        shape_f = lon_f.shape

        num_peri_dims = 1 if periodic else None
        periodic_dim = 0 if periodic else None
        pole_dim = 1 if periodic else None

        lat_b, lon_b = _get_grid_bounds(ds)

        if (lat_b is None or lon_b is None) and method == "conservative":
            try:
                ds_with_bounds = ds.cf.add_bounds(["latitude", "longitude"])
                lat_b, lon_b = _get_grid_bounds(ds_with_bounds)
            except Exception:
                pass

        has_bounds = lat_b is not None and lon_b is not None

        if method == "conservative" and not has_bounds:
            raise ValueError("Conservative regridding requires cell boundaries.")

        staggerlocs = [esmpy.StaggerLoc.CENTER]
        if has_bounds:
            staggerlocs.append(esmpy.StaggerLoc.CORNER)

        grid = esmpy.Grid(
            np.array(shape_f),
            staggerloc=staggerlocs,
            coord_sys=esmpy.CoordSys.SPH_DEG,
            num_peri_dims=num_peri_dims,
            periodic_dim=periodic_dim,
            pole_dim=pole_dim,
        )

        grid_lon = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)
        grid_lat = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)
        grid_lon[...] = lon_f.astype(np.float64)
        grid_lat[...] = lat_f.astype(np.float64)

        if has_bounds:
            if lon_b.ndim == 1 and lat_b.ndim == 1:
                lon_b_vals, lat_b_vals = np.meshgrid(lon_b, lat_b)
            else:
                lon_b_vals, lat_b_vals = lon_b, lat_b

            grid_lon_b = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)
            grid_lat_b = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)

            lon_b_vals_f = lon_b_vals.T
            lat_b_vals_f = lat_b_vals.T

            if periodic:
                lon_b_vals_f = lon_b_vals_f[:-1, :]
                lat_b_vals_f = lat_b_vals_f[:-1, :]

            grid_lon_b[...] = lon_b_vals_f.astype(np.float64)
            grid_lat_b[...] = lat_b_vals_f.astype(np.float64)

        if is_source and mask_var and mask_var in ds:
            grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
            mask_ptr = grid.get_item(
                esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER
            )
            mask_f = ds[mask_var].values.T
            mask_ptr[...] = mask_f.astype(np.int32)
        return grid


class Regridder:
    """
    Optimized ESMF-based regridder for xarray DataArrays and Datasets.

    This regridder supports both eager (NumPy) and lazy (Dask) backends.
    It uses ESMPy to generate weights and applies them using xarray.apply_ufunc.

    Attributes
    ----------
    source_grid_ds : xr.Dataset
        The source grid dataset containing 'lat' and 'lon'.
    target_grid_ds : xr.Dataset
        The target grid dataset containing 'lat' and 'lon'.
    method : str
        The regridding method (e.g., 'bilinear', 'conservative').
    mask_var : str, optional
        The variable name in source_grid_ds to use as a mask.
    filename : str
        The path to save/load weights.
    skipna : bool
        Whether to handle NaNs by re-normalizing weights.
    na_thres : float
        Threshold for NaN handling.
    periodic : bool
        Whether the grid is periodic in longitude.
    """

    def __init__(
        self,
        source_grid_ds: xr.Dataset,
        target_grid_ds: xr.Dataset,
        method: str = "bilinear",
        mask_var: Optional[str] = None,
        reuse_weights: bool = False,
        filename: str = "weights.nc",
        skipna: bool = False,
        na_thres: float = 1.0,
        periodic: bool = False,
        mpi: bool = False,
        parallel: bool = False,
        extrap_method: Optional[str] = None,
        extrap_dist_exponent: float = 2.0,
    ) -> None:
        """
        Initialize the Regridder.

        Parameters
        ----------
        source_grid_ds : xr.Dataset
            Contain 'lat' and 'lon'.
        target_grid_ds : xr.Dataset
            Contain 'lat' and 'lon'.
        method : str, default 'bilinear'
            Regridding method (bilinear, conservative, nearest_s2d, nearest_d2s, patch).
        mask_var : str, optional
            Variable name for mask (1=valid, 0=masked).
        reuse_weights : bool, default False
            Load weights from filename if it exists.
        filename : str, default 'weights.nc'
            Path to weights file.
        skipna : bool, default False
            Handle NaNs in input data by re-normalizing weights.
        na_thres : float, default 1.0
            Threshold for NaN handling.
        periodic : bool, default False
            Whether the grid is periodic in longitude.
        mpi : bool, default False
            Whether to use MPI for parallel weight generation.
        parallel : bool, default False
            Whether to use Dask for parallel weight generation.
        extrap_method : str, optional
            Extrapolation method (nearest_s2d, nearest_idw, creep_fill).
        extrap_dist_exponent : float, default 2.0
            Exponent for IDW extrapolation.
        """
        if esmpy is None:
            raise ImportError("ESMPy is required for Regridder.")

        self._manager = esmpy.Manager(debug=False)

        self.source_grid_ds = source_grid_ds
        self.target_grid_ds = target_grid_ds
        self.method = method
        self.mask_var = mask_var
        self.filename = filename
        self.skipna = skipna
        self.na_thres = na_thres
        self.periodic = periodic
        self.parallel = parallel
        self.extrap_method = extrap_method
        self.extrap_dist_exponent = extrap_dist_exponent

        self.method_map = {
            "bilinear": esmpy.RegridMethod.BILINEAR,
            "conservative": esmpy.RegridMethod.CONSERVE,
            "nearest_s2d": esmpy.RegridMethod.NEAREST_STOD,
            "nearest_d2s": esmpy.RegridMethod.NEAREST_DTOS,
            "patch": esmpy.RegridMethod.PATCH,
        }

        self.extrap_method_map = {
            "nearest_s2d": esmpy.ExtrapMethod.NEAREST_STOD,
            "nearest_idw": esmpy.ExtrapMethod.NEAREST_IDAVG,
            "creep_fill": esmpy.ExtrapMethod.CREEP_FILL,
        }

        # Internal state
        self._shape_source: Optional[Tuple[int, ...]] = None
        self._shape_target: Optional[Tuple[int, ...]] = None
        self._dims_source: Optional[Tuple[str, ...]] = None
        self._dims_target: Optional[Tuple[str, ...]] = None
        self._is_unstructured_src: bool = False
        self._is_unstructured_tgt: bool = False
        self._total_weights: Optional[np.ndarray] = None
        self._weights_matrix: Optional[coo_matrix] = None
        self._loaded_method: Optional[str] = None
        self._loaded_periodic: Optional[bool] = None
        self._loaded_extrap: Optional[str] = None
        self.generation_time: Optional[float] = None

        if reuse_weights and os.path.exists(filename):
            self._load_weights()
            self._validate_weights()
        else:
            if parallel:
                self._generate_weights_parallel()
            else:
                self._generate_weights()
            if reuse_weights:
                self._save_weights()

    def _validate_weights(self) -> None:
        """Validate loaded weights against provided grids."""
        _, _, src_shape, src_dims, _ = _get_mesh_info(self.source_grid_ds)
        _, _, dst_shape, dst_dims, _ = _get_mesh_info(self.target_grid_ds)

        if src_shape != self._shape_source:
            raise ValueError(
                f"Source grid shape mismatch: {src_shape} vs {self._shape_source}"
            )
        if dst_shape != self._shape_target:
            raise ValueError(
                f"Target grid shape mismatch: {dst_shape} vs {self._shape_target}"
            )
        if self._loaded_method is not None and self._loaded_method != self.method:
            raise ValueError(f"Method mismatch: {self.method} vs {self._loaded_method}")
        if self._loaded_periodic is not None and self._loaded_periodic != self.periodic:
            raise ValueError(
                f"Periodic mismatch: {self.periodic} vs {self._loaded_periodic}"
            )

    def _generate_weights(self) -> None:
        """Generate regridding weights using ESMPy (Serial/MPI)."""
        start_time = time.perf_counter()
        src_obj = _create_esmf_grid(
            self.source_grid_ds,
            method=self.method,
            periodic=self.periodic,
            mask_var=self.mask_var,
            is_source=True,
        )
        dst_obj = _create_esmf_grid(
            self.target_grid_ds,
            method=self.method,
            periodic=self.periodic,
            is_source=False,
        )

        # Update internal state from helper results
        _, _, self._shape_source, self._dims_source, self._is_unstructured_src = (
            _get_mesh_info(self.source_grid_ds)
        )
        _, _, self._shape_target, self._dims_target, self._is_unstructured_tgt = (
            _get_mesh_info(self.target_grid_ds)
        )

        src_field = esmpy.Field(src_obj, name="src")
        dst_field = esmpy.Field(dst_obj, name="dst")

        regrid_kwargs = {
            "regrid_method": self.method_map[self.method],
            "unmapped_action": esmpy.UnmappedAction.IGNORE,
            "factors": True,
        }

        if self.extrap_method and self.extrap_method != "none":
            regrid_kwargs["extrap_method"] = self.extrap_method_map[self.extrap_method]
            regrid_kwargs["extrap_dist_exponent"] = self.extrap_dist_exponent

        if not self._is_unstructured_src and not self._is_unstructured_tgt:
            if self.mask_var and self.mask_var in self.source_grid_ds:
                regrid_kwargs["src_mask_values"] = np.array([0], dtype=np.int32)

        regrid = esmpy.Regrid(src_field, dst_field, **regrid_kwargs)
        weights = regrid.get_weights_dict(deep_copy=True)

        pet_count = esmpy.pet_count()
        local_pet = esmpy.local_pet()

        if pet_count > 1:
            try:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                all_weights = comm.gather(weights, root=0)
                if local_pet == 0:
                    rows = np.concatenate([w["row_dst"] for w in all_weights]) - 1
                    cols = np.concatenate([w["col_src"] for w in all_weights]) - 1
                    data = np.concatenate([w["weights"] for w in all_weights])
                else:
                    rows, cols, data = np.array([]), np.array([]), np.array([])
            except ImportError:
                rows, cols, data = (
                    weights["row_dst"] - 1,
                    weights["col_src"] - 1,
                    weights["weights"],
                )
        else:
            rows, cols, data = (
                weights["row_dst"] - 1,
                weights["col_src"] - 1,
                weights["weights"],
            )

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        if len(rows) > 0:
            self._weights_matrix = coo_matrix(
                (data, (rows, cols)), shape=(n_dst, n_src)
            ).tocsr()

        if self.skipna and self._weights_matrix is not None:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

        self.generation_time = time.perf_counter() - start_time

    def _generate_weights_parallel(self) -> None:
        """Generate weights in parallel using Dask 2D decomposition."""
        if dask is None:
            raise ImportError("Dask and distributed are required for parallel=True.")

        start_time = time.perf_counter()

        # Get mesh info without .values (Scientific Hygiene)
        (
            lon_s,
            lat_s,
            self._shape_source,
            self._dims_source,
            self._is_unstructured_src,
        ) = _get_mesh_info(self.source_grid_ds)
        (
            lon_t,
            lat_t,
            self._shape_target,
            self._dims_target,
            self._is_unstructured_tgt,
        ) = _get_mesh_info(self.target_grid_ds)

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        # 2D Decomposition of target grid
        if hasattr(self.target_grid_ds, "chunks") and self.target_grid_ds.chunks:
            chunks = self.target_grid_ds.chunks
        else:
            chunks = {
                self._dims_target[0]: (
                    self._shape_target[0] // 2,
                    self._shape_target[0] - self._shape_target[0] // 2,
                ),
                self._dims_target[1]: (
                    self._shape_target[1] // 2,
                    self._shape_target[1] - self._shape_target[1] // 2,
                ),
            }

        def get_chunk_slices(chunks_dict):
            dims = self._dims_target
            slices = []
            start_i = 0
            for size_i in chunks_dict[dims[0]]:
                start_j = 0
                for size_j in chunks_dict[dims[1]]:
                    slices.append(
                        {
                            dims[0]: slice(start_i, start_i + size_i),
                            dims[1]: slice(start_j, start_j + size_j),
                        }
                    )
                    start_j += size_j
                start_i += size_i
            return slices

        slices = get_chunk_slices(chunks)

        try:
            client = get_client()
            src_min = (
                self.source_grid_ds[[self.mask_var]]
                if self.mask_var
                else self.source_grid_ds[[]]
            )
            src_min = src_min.assign_coords(
                {
                    c: self.source_grid_ds.coords[c]
                    for c in self.source_grid_ds.coords
                    if any(
                        d in self._dims_source
                        for d in self.source_grid_ds.coords[c].dims
                    )
                }
            ).compute()
            src_future = client.scatter(src_min)
        except Exception:
            src_future = self.source_grid_ds.compute()

        delayed_weights = [
            dask.delayed(_compute_chunk_weights)(
                src_future,
                self.target_grid_ds.isel(slc).compute(),
                slc,
                self._shape_target,
                self.method,
                self.periodic,
                self.mask_var,
                self.extrap_method,
                self.extrap_dist_exponent,
            )
            for slc in slices
        ]

        all_weights = dask.compute(*delayed_weights)

        rows = np.concatenate([w[0] for w in all_weights])
        cols = np.concatenate([w[1] for w in all_weights])
        data = np.concatenate([w[2] for w in all_weights])

        if len(rows) > 0:
            self._weights_matrix = coo_matrix(
                (data, (rows, cols)), shape=(n_dst, n_src)
            ).tocsr()

        if self.skipna and self._weights_matrix is not None:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

        self.generation_time = time.perf_counter() - start_time

    def _save_weights(self) -> None:
        """Save weights to a NetCDF file."""
        if esmpy.local_pet() != 0 or self._weights_matrix is None:
            return

        weights_coo = self._weights_matrix.tocoo()
        ds_weights = xr.Dataset(
            data_vars={
                "row": (["n_s"], weights_coo.row + 1),
                "col": (["n_s"], weights_coo.col + 1),
                "S": (["n_s"], weights_coo.data),
            },
            attrs={
                "n_src": self._weights_matrix.shape[1],
                "n_dst": self._weights_matrix.shape[0],
                "shape_src": list(self._shape_source) if self._shape_source else [],
                "shape_dst": list(self._shape_target) if self._shape_target else [],
                "dims_src": list(self._dims_source) if self._dims_source else [],
                "dims_target": list(self._dims_target) if self._dims_target else [],
                "is_unstructured_src": int(self._is_unstructured_src),
                "is_unstructured_tgt": int(self._is_unstructured_tgt),
                "method": self.method,
                "periodic": int(self.periodic),
                "parallel": int(self.parallel),
                "extrap_method": self.extrap_method or "none",
                "extrap_dist_exponent": self.extrap_dist_exponent,
                "generation_time": self.generation_time or 0.0,
            },
        )
        update_history(ds_weights, "Weights generated by Regridder")
        ds_weights.to_netcdf(self.filename)

    def _load_weights(self) -> None:
        """Load weights from a NetCDF file."""
        with xr.open_dataset(self.filename) as ds_weights:
            ds_weights.load()
            rows = ds_weights["row"].values - 1
            cols = ds_weights["col"].values - 1
            data = ds_weights["S"].values
            n_src = ds_weights.attrs["n_src"]
            n_dst = ds_weights.attrs["n_dst"]

            def _to_tuple(attr: Any) -> Tuple[Any, ...]:
                if isinstance(attr, str):
                    attr = attr.strip("()[]").replace(" ", "").split(",")
                    return tuple(int(x) if x.isdigit() else x for x in attr if x)
                return tuple(attr)

            self._shape_source = _to_tuple(ds_weights.attrs["shape_src"])
            self._shape_target = _to_tuple(ds_weights.attrs["shape_dst"])
            self._dims_source = _to_tuple(ds_weights.attrs["dims_src"])
            self._dims_target = _to_tuple(ds_weights.attrs["dims_target"])
            self._is_unstructured_src = bool(
                ds_weights.attrs.get("is_unstructured_src", False)
            )
            self._is_unstructured_tgt = bool(
                ds_weights.attrs.get("is_unstructured_tgt", False)
            )
            self._loaded_method = ds_weights.attrs.get("method")
            self._loaded_periodic = bool(ds_weights.attrs.get("periodic", False))
            self._loaded_extrap = ds_weights.attrs.get("extrap_method", "none")
            self.generation_time = ds_weights.attrs.get("generation_time")

        self._weights_matrix = coo_matrix(
            (data, (rows, cols)), shape=(n_dst, n_src)
        ).tocsr()

        if self.skipna:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

    def __repr__(self) -> str:
        """
        String representation of the Regridder.

        Returns
        -------
        str
            Summary of the regridder configuration.
        """
        return (
            f"Regridder(method={self.method}, "
            f"src_shape={self._shape_source}, "
            f"dst_shape={self._shape_target}, "
            f"periodic={self.periodic}, "
            f"parallel={self.parallel})"
        )

    def __call__(
        self, obj: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        if isinstance(obj, xr.Dataset):
            return self._regrid_dataset(obj)
        elif isinstance(obj, xr.DataArray):
            return self._regrid_dataarray(obj)
        else:
            raise TypeError("Input must be an xarray.DataArray or xarray.Dataset.")

    def _regrid_dataarray(
        self, da_in: xr.DataArray, update_history_attr: bool = True
    ) -> xr.DataArray:
        def _apply_weights(data_block: np.ndarray) -> np.ndarray:
            original_shape = data_block.shape
            n_spatial = int(np.prod(self._shape_source))
            n_other = int(np.prod(original_shape[: -len(self._dims_source)]))
            flat_data = data_block.reshape(n_other, n_spatial)

            if self.skipna:
                mask = np.isnan(flat_data)
                has_nans = np.any(mask)
                if not has_nans:
                    result = (self._weights_matrix @ flat_data.T).T
                    if self._total_weights is not None:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            result = result / self._total_weights
                else:
                    safe_data = np.where(mask, 0.0, flat_data)
                    result = (self._weights_matrix @ safe_data.T).T
                    weights_sum = (
                        self._weights_matrix @ (~mask).astype(np.float32).T
                    ).T
                    with np.errstate(divide="ignore", invalid="ignore"):
                        final_result = result / weights_sum
                        if self._total_weights is not None:
                            fraction_valid = weights_sum / self._total_weights
                            final_result = np.where(
                                fraction_valid >= (1.0 - self.na_thres - 1e-6),
                                final_result,
                                np.nan,
                            )
                    result = final_result
            else:
                result = (self._weights_matrix @ flat_data.T).T

            return result.reshape(
                original_shape[: -len(self._dims_source)] + self._shape_target
            )

        temp_output_core_dims = [f"{d}_regridded" for d in self._dims_target]

        out = xr.apply_ufunc(
            _apply_weights,
            da_in,
            input_core_dims=[list(self._dims_source)],
            output_core_dims=[temp_output_core_dims],
            dask="parallelized",
            output_dtypes=[da_in.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {
                    d: s for d, s in zip(temp_output_core_dims, self._shape_target)
                },
                "allow_rechunk": True,
            },
        )

        out = out.rename(
            {temp: orig for temp, orig in zip(temp_output_core_dims, self._dims_target)}
        )

        out.name = da_in.name
        out.attrs.update(da_in.attrs)
        out = out.assign_coords(
            {
                c: self.target_grid_ds.coords[c]
                for c in self.target_grid_ds.coords
                if set(self.target_grid_ds.coords[c].dims).issubset(
                    set(self._dims_target)
                )
            }
        )

        if update_history_attr:
            history_msg = f"Regridded using Regridder (method={self.method})"
            if self.generation_time:
                history_msg += f". Weight generation time: {self.generation_time:.4f}s"
            update_history(out, history_msg)
        return out

    def _regrid_dataset(self, ds_in: xr.Dataset) -> xr.Dataset:
        regridded_vars = {
            name: self._regrid_dataarray(da, False)
            if all(d in da.dims for d in self._dims_source)
            else da
            for name, da in ds_in.data_vars.items()
        }
        out = xr.Dataset(regridded_vars, attrs=ds_in.attrs)
        for c in ds_in.coords:
            if c not in out.coords and not any(
                d in self._dims_source for d in ds_in.coords[c].dims
            ):
                out = out.assign_coords({c: ds_in.coords[c]})

        history_msg = f"Regridded Dataset using Regridder (method={self.method})"
        if self.generation_time:
            history_msg += f". Weight generation time: {self.generation_time:.4f}s"
        update_history(out, history_msg)
        return out
