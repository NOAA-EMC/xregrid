from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import xarray as xr

from xregrid.utils import get_crs_info, _find_coord

if TYPE_CHECKING:
    from xregrid.regridder import Regridder

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    import hvplot.xarray  # noqa: F401
    import holoviews as hv
except ImportError:
    hvplot = None
    hv = None
else:
    hvplot = True


def plot_static(
    da: xr.DataArray,
    projection: Any = None,
    transform: Any = None,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track A: Publication-quality static plot using Matplotlib and Cartopy.

    Parameters
    ----------
    da : xr.DataArray
        The 2D DataArray to plot.
    projection : cartopy.crs.Projection, optional
        The projection to use for the axes. Defaults to ccrs.PlateCarree() if cartopy is available.
    transform : cartopy.crs.Projection, optional
        The transform to use for the plot call. Defaults to ccrs.PlateCarree() if cartopy is available.
    title : str, optional
        The plot title.
    **kwargs : Any
        Additional arguments passed to da.plot().

    Returns
    -------
    Any
        The plot object (e.g., matplotlib QuadMesh or FacetGrid).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    if plt is None:
        raise ImportError(
            "Matplotlib is required for plot_static. "
            "Install it with `pip install matplotlib`."
        )

    # Handle axes and faceting early to avoid multiple 'ax' arguments
    ax = kwargs.pop("ax", None)
    is_faceted = "col" in kwargs or "row" in kwargs

    # No Ambiguous Plots.
    # Identify spatial and faceting dimensions to slice away everything else.
    # We do this early so it applies even if cartopy is missing.

    # Identify spatial dimensions using cf-xarray or fallbacks for robust slicing
    lat_da = None
    lon_da = None
    try:
        # Use enhanced discovery
        lat_da = _find_coord(da, "latitude")
        lon_da = _find_coord(da, "longitude")

        if lat_da is not None and lon_da is not None:
            spatial_dims = set(lat_da.dims) | set(lon_da.dims)
        else:
            # Fallback to cf-xarray directly
            lat_dims = da.cf["latitude"].dims
            lon_dims = da.cf["longitude"].dims
            spatial_dims = set(lat_dims) | set(lon_dims)
    except (KeyError, AttributeError, ImportError, ValueError):
        # Fallback to assuming the last two dimensions are spatial
        spatial_dims = set(da.dims[-2:])

    # Identify dimensions used for faceting
    facet_dims = {kwargs.get("col"), kwargs.get("row")} - {None}

    # Dimensions that are neither spatial nor used for faceting
    extra_dims = [d for d in da.dims if d not in spatial_dims and d not in facet_dims]

    if extra_dims:
        first_slice = {d: 0 for d in extra_dims}
        warnings.warn(
            f"DataArray has {da.ndim} dimensions, but only 2 spatial dimensions "
            f"(plus optional faceting) are supported for static plots. "
            f"Automatically selecting the first slice along {extra_dims}: {first_slice}. "
            "To plot other slices, subset your data before calling plot_static."
        )
        da = da.isel(first_slice)

    if ccrs is None:
        # Fallback to standard matplotlib if cartopy is missing
        if ax is None:
            ax = plt.gca()
        im = da.plot(ax=ax, **kwargs)
        if title:
            ax.set_title(title)
        return im

    if transform is None and ccrs is not None:
        proj_crs = get_crs_info(da)

        if proj_crs:
            try:
                # Map pyproj CRS to Cartopy projections
                if proj_crs.is_geographic:
                    transform = ccrs.PlateCarree()
                elif proj_crs.is_projected:
                    # Attempt robust projection detection
                    # UTM detection
                    if proj_crs.utm_zone:
                        transform = ccrs.UTM(
                            zone=int(proj_crs.utm_zone[:-1]),
                            southern_hemisphere="S" in proj_crs.utm_zone,
                        )
                    # Mercator
                    elif "merc" in proj_crs.to_dict().get("proj", ""):
                        transform = ccrs.Mercator()
                    # Lambert Conformal
                    elif "lcc" in proj_crs.to_dict().get("proj", ""):
                        transform = ccrs.LambertConformal(
                            central_longitude=proj_crs.to_dict().get("lon_0", 0.0),
                            central_latitude=proj_crs.to_dict().get("lat_0", 0.0),
                        )
            except Exception:
                pass

    if projection is None:
        projection = ccrs.PlateCarree()
    if transform is None:
        transform = ccrs.PlateCarree()

    if ax is not None:
        if is_faceted:
            warnings.warn(
                "Providing an 'ax' with faceting ('col' or 'row') is not supported by xarray and will be ignored."
            )
            ax = None
        else:
            # Ensure the existing axes is a GeoAxes if we are using cartopy
            is_geoaxes = False
            try:
                import cartopy.mpl.geoaxes as geoaxes

                is_geoaxes = isinstance(ax, geoaxes.GeoAxes)
            except ImportError:
                is_geoaxes = hasattr(ax, "projection")

            if not is_geoaxes:
                warnings.warn(
                    "The provided axes does not appear to be a Cartopy GeoAxes. "
                    "Geospatial plotting may not work as expected. "
                    "Ensure your axes was created with a projection (e.g., plt.axes(projection=...))."
                )

    if ax is None and not is_faceted:
        # Strictly enforce projection in axes creation
        if projection is None and ccrs is not None:
            projection = ccrs.PlateCarree()
        ax = plt.axes(projection=projection)

    # Enforce transform for geospatial accuracy
    if transform is None and ccrs is not None:
        transform = ccrs.PlateCarree()

    if "transform" not in kwargs:
        kwargs["transform"] = transform

    if is_faceted and "subplot_kws" not in kwargs:
        kwargs["subplot_kws"] = {"projection": projection}

    if da.ndim == 1 and lat_da is not None and lon_da is not None:
        # For unstructured 1D data, use scatter to ensure geospatial representation
        # Following Aero "No Ambiguous Plots" rule.
        if "x" not in kwargs:
            kwargs["x"] = lon_da.name
        if "y" not in kwargs:
            kwargs["y"] = lat_da.name
        im = da.plot.scatter(ax=ax, **kwargs)
    else:
        im = da.plot(ax=ax, **kwargs)

    if is_faceted:
        # im is a FacetGrid
        if hasattr(im, "axes"):
            for a in im.axes.flat:
                if hasattr(a, "coastlines"):
                    a.coastlines()
        if title:
            plt.suptitle(title, y=1.02)
    else:
        # im is a QuadMesh or similar
        if hasattr(ax, "coastlines"):
            ax.coastlines()

        if title is None:
            title = da.name if da.name else "Static Map"
        ax.set_title(title)

    return im


def plot(
    da: xr.DataArray,
    mode: str = "static",
    **kwargs: Any,
) -> Any:
    """
    Unified entry point for xregrid plotting following the Two-Track Rule.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to plot.
    mode : str, default 'static'
        The plotting mode: 'static' (Track A: Publication) or
        'interactive' (Track B: Exploration).
    **kwargs : Any
        Additional arguments passed to plot_static or plot_interactive.

    Returns
    -------
    Any
        The plot object (Matplotlib artist or HvPlot object).

    Raises
    ------
    ValueError
        If an unknown plotting mode is provided.
    """
    if mode == "static":
        return plot_static(da, **kwargs)
    elif mode == "interactive":
        return plot_interactive(da, **kwargs)
    else:
        raise ValueError(
            f"Unknown plotting mode: '{mode}'. Must be 'static' or 'interactive'."
        )


def plot_interactive(
    da: xr.DataArray,
    rasterize: bool = True,
    title: str = "Interactive Map",
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive plot using HvPlot.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to plot.
    rasterize : bool, default True
        Whether to rasterize the grid for large datasets.
    title : str, default 'Interactive Map'
        The plot title.
    **kwargs : Any
        Additional arguments passed to da.hvplot().

    Returns
    -------
    Any
        The interactive plot object (HvPlot/HoloViews).

    Raises
    ------
    ImportError
        If HvPlot is not installed.
    """
    if not hvplot:
        raise ImportError(
            "HvPlot is required for plot_interactive. "
            "Install it with `pip install hvplot`."
        )

    # Automated CRS discovery for Track B (Interactive)
    # This ensures "No Ambiguous Plots" even in exploratory mode.
    if "geo" not in kwargs:
        crs_obj = get_crs_info(da)
        if crs_obj:
            kwargs["geo"] = True

    # Aero Protocol: Ensure 1D unstructured grids are rendered as maps, not line plots.
    if da.ndim == 1 and "kind" not in kwargs:
        lat_da = _find_coord(da, "latitude")
        lon_da = _find_coord(da, "longitude")
        if lat_da is not None and lon_da is not None:
            kwargs["kind"] = "points"
            if "x" not in kwargs:
                kwargs["x"] = lon_da.name
            if "y" not in kwargs:
                kwargs["y"] = lat_da.name

    return da.hvplot(rasterize=rasterize, title=title, **kwargs)


def plot_diagnostics(
    regridder: "Regridder",
    projection: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Track A: Plot spatial diagnostics for a Regridder.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance to diagnose.
    projection : Any, optional
        The projection for the axes. Defaults to ccrs.PlateCarree() if available.
    **kwargs : Any
        Additional arguments passed to plot_static.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    """
    if plt is None:
        raise ImportError("Matplotlib is required for plot_diagnostics.")

    # Automated projection discovery (No Ambiguous Plots)
    if projection is None and ccrs is not None:
        # Attempt to discover projection from target grid
        target_crs = get_crs_info(regridder.target_grid_ds)
        if target_crs:
            if target_crs.is_geographic:
                projection = ccrs.PlateCarree()
            elif target_crs.is_projected:
                # Basic mapping to common projections
                if target_crs.utm_zone:
                    projection = ccrs.UTM(
                        zone=int(target_crs.utm_zone[:-1]),
                        southern_hemisphere="S" in target_crs.utm_zone,
                    )
                elif "merc" in target_crs.to_dict().get("proj", ""):
                    projection = ccrs.Mercator()
                else:
                    projection = ccrs.PlateCarree()
        else:
            projection = ccrs.PlateCarree()

    ds_diag = regridder.diagnostics()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        subplot_kw={"projection": projection},
    )

    plot_static(
        ds_diag.weight_sum,
        ax=axes[0],
        title="Weight Sum",
        cmap="viridis",
        **kwargs,
    )

    plot_static(
        ds_diag.unmapped_mask,
        ax=axes[1],
        title="Unmapped Mask (1=Unmapped)",
        cmap="Reds",
        **kwargs,
    )

    fig.suptitle(f"Regridder Diagnostics ({regridder.method})", fontsize=16)
    plt.tight_layout()

    return fig


def plot_diagnostics_interactive(
    regridder: "Regridder",
    rasterize: bool = True,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive diagnostic plot.

    Uses HvPlot and HoloViews to provide a side-by-side interactive view
    of weight_sum and unmapped_mask.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance to diagnose.
    rasterize : bool, default True
        Whether to rasterize the grid for large datasets.
    title : str, optional
        Overall plot title.
    **kwargs : Any
        Additional arguments passed to hvplot calls.

    Returns
    -------
    Any
        The composed HoloViews object (Layout).

    Raises
    ------
    ImportError
        If HvPlot or HoloViews is not installed.
    """
    if not hvplot or hv is None:
        raise ImportError(
            "HvPlot and HoloViews are required for plot_diagnostics_interactive. "
            "Install them with `pip install hvplot holoviews`."
        )

    ds_diag = regridder.diagnostics()

    # 1. Weight Sum Plot
    p_sum = ds_diag.weight_sum.hvplot(
        rasterize=rasterize, cmap="viridis", title="Weight Sum", **kwargs
    )

    # 2. Unmapped Mask Plot
    p_mask = ds_diag.unmapped_mask.hvplot(
        rasterize=rasterize, cmap="Reds", title="Unmapped Mask (1=Unmapped)", **kwargs
    )

    layout = (p_sum + p_mask).cols(2)

    if title is None:
        title = f"Regridder Diagnostics ({regridder.method})"

    layout = layout.opts(title=title)

    return layout


def plot_comparison(
    da_src: xr.DataArray,
    da_tgt: xr.DataArray,
    regridder: Optional[Any] = None,
    projection: Any = None,
    transform: Any = None,
    cmap: str = "viridis",
    diff_cmap: str = "RdBu_r",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track A: Publication-quality comparison plot (Source, Target, Difference).

    Parameters
    ----------
    da_src : xr.DataArray
        The source DataArray.
    da_tgt : xr.DataArray
        The target (regridded) DataArray.
    regridder : Regridder, optional
        The regridder used to transform da_src to da_tgt.
        If provided, it will be used to calculate the difference plot correctly.
    projection : Any, optional
        The projection for the axes.
    transform : Any, optional
        The transform for the plot call.
    cmap : str, default 'viridis'
        Colormap for the data plots.
    diff_cmap : str, default 'RdBu_r'
        Colormap for the difference plot.
    title : str, optional
        Overall figure title.
    **kwargs : Any
        Additional arguments passed to plot_static.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    """
    if plt is None:
        raise ImportError("Matplotlib is required for plot_comparison.")

    if projection is None and ccrs is not None:
        projection = ccrs.PlateCarree()

    # Enforce projection on all subplots for comparison consistency
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        subplot_kw={"projection": projection},
    )

    # 1. Source Plot
    plot_static(
        da_src,
        ax=axes[0],
        projection=projection,
        transform=transform,
        cmap=cmap,
        title="Source Grid",
        **kwargs,
    )

    # 2. Target Plot
    plot_static(
        da_tgt,
        ax=axes[1],
        projection=projection,
        transform=transform,
        cmap=cmap,
        title="Target Grid",
        **kwargs,
    )

    # 3. Difference Plot
    # Use Regridder if provided for exact difference, otherwise fallback to interp_like
    try:
        if regridder is not None:
            da_src_interp = regridder(da_src)
        else:
            da_src_interp = da_src.interp_like(da_tgt, method="linear")

        diff = da_tgt - da_src_interp
        plot_static(
            diff,
            ax=axes[2],
            projection=projection,
            transform=transform,
            cmap=diff_cmap,
            title="Difference (Tgt - Src_interp)",
            **kwargs,
        )
    except Exception as e:
        axes[2].text(
            0.5,
            0.5,
            f"Could not compute difference:\n{e}",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Difference")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    return fig


def plot_comparison_interactive(
    da_src: xr.DataArray,
    da_tgt: xr.DataArray,
    regridder: Optional[Any] = None,
    rasterize: bool = True,
    cmap: str = "viridis",
    diff_cmap: str = "RdBu_r",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive comparison plot (Source, Target, Difference).

    Uses HvPlot and HoloViews to provide a side-by-side interactive view.

    Parameters
    ----------
    da_src : xr.DataArray
        The source DataArray.
    da_tgt : xr.DataArray
        The target (regridded) DataArray.
    regridder : Regridder, optional
        The regridder used to transform da_src to da_tgt.
        If provided, it will be used to calculate the difference plot correctly.
    rasterize : bool, default True
        Whether to rasterize the grid for large datasets.
    cmap : str, default 'viridis'
        Colormap for the data plots.
    diff_cmap : str, default 'RdBu_r'
        Colormap for the difference plot.
    title : str, optional
        Overall plot title.
    **kwargs : Any
        Additional arguments passed to hvplot calls.

    Returns
    -------
    Any
        The composed HoloViews object (Layout).

    Raises
    ------
    ImportError
        If HvPlot or HoloViews is not installed.
    """
    if not hvplot or hv is None:
        raise ImportError(
            "HvPlot and HoloViews are required for plot_comparison_interactive. "
            "Install them with `pip install hvplot holoviews`."
        )

    # 1. Source Plot
    p_src = da_src.hvplot(rasterize=rasterize, cmap=cmap, title="Source Grid", **kwargs)

    # 2. Target Plot
    p_tgt = da_tgt.hvplot(rasterize=rasterize, cmap=cmap, title="Target Grid", **kwargs)

    # 3. Difference Plot
    try:
        if regridder is not None:
            da_src_interp = regridder(da_src)
        else:
            da_src_interp = da_src.interp_like(da_tgt, method="linear")

        diff = da_tgt - da_src_interp
        p_diff = diff.hvplot(
            rasterize=rasterize,
            cmap=diff_cmap,
            title="Difference (Tgt - Src_interp)",
            **kwargs,
        )
    except Exception as e:
        # Fallback to a placeholder if difference computation fails
        p_diff = hv.Text(0.5, 0.5, f"Could not compute difference:\n{e}")

    layout = (p_src + p_tgt + p_diff).cols(3)

    if title:
        layout = layout.opts(title=title)

    return layout


def plot_weights(
    regridder: "Regridder",
    row_idx: int,
    mode: str = "static",
    **kwargs: Any,
) -> Any:
    """
    Visualize source points contributing to a specific destination point.

    Two-Track Rule:
    - mode='static' (Track A): Publication-quality plot using Matplotlib/Cartopy.
    - mode='interactive' (Track B): Exploratory plot using HvPlot/HoloViews.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance.
    row_idx : int
        The index of the destination point (0-based).
    mode : str, default 'static'
        The plotting mode: 'static' or 'interactive'.
    **kwargs : Any
        Additional arguments passed to the plotting functions.

    Returns
    -------
    Any
        The plot object.
    """
    if mode == "static":
        return _plot_weights_static(regridder, row_idx, **kwargs)
    elif mode == "interactive":
        rasterize = kwargs.pop("rasterize", True)
        return plot_weights_interactive(
            regridder, row_idx, rasterize=rasterize, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown plotting mode: '{mode}'. Must be 'static' or 'interactive'."
        )


def _plot_weights_static(
    regridder: "Regridder",
    row_idx: int,
    **kwargs: Any,
) -> Any:
    """
    Track A: Visualize source points contributing to a specific destination point.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance.
    row_idx : int
        The index of the destination point (0-based).
    **kwargs : Any
        Additional arguments passed to plot_static.

    Returns
    -------
    Any
        The plot object.
    """
    da_weights = _get_weight_row_da(regridder, row_idx)
    return plot_static(
        da_weights, title=f"Weights for Destination Point {row_idx}", **kwargs
    )


def plot_weights_interactive(
    regridder: "Regridder",
    row_idx: int,
    rasterize: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive visualization of weights for a destination point.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance.
    row_idx : int
        The index of the destination point (0-based).
    rasterize : bool, default True
        Whether to rasterize the grid for large datasets.
    **kwargs : Any
        Additional arguments passed to plot_interactive.

    Returns
    -------
    Any
        The interactive plot object.
    """
    da_weights = _get_weight_row_da(regridder, row_idx)
    return plot_interactive(
        da_weights,
        rasterize=rasterize,
        title=f"Weights for Destination Point {row_idx}",
        **kwargs,
    )


def _get_weight_row_da(regridder: "Regridder", row_idx: int) -> xr.DataArray:
    """
    Extract a single weight row as a DataArray, optimized for remote weights.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance.
    row_idx : int
        The index of the destination point.

    Returns
    -------
    xr.DataArray
        The weights on the source grid.
    """
    if hasattr(regridder._weights_matrix, "key"):
        # Optimized Distributed Path: extract row on cluster
        from .parallel import _get_weight_row_task

        row = regridder._dask_client.submit(
            _get_weight_row_task, regridder._weights_matrix, row_idx
        ).result()
    else:
        # Eager Path
        matrix = regridder.weights
        row = matrix.getrow(row_idx).toarray().flatten()

    # Reconstruct 2D/1D array on source grid
    coords = {
        c: regridder.source_grid_ds.coords[c]
        for c in regridder.source_grid_ds.coords
        if regridder._dims_source is not None
        and set(regridder.source_grid_ds.coords[c].dims).issubset(
            set(regridder._dims_source)
        )
    }

    # Include topology/mapping from source grid
    for v in regridder.source_grid_ds.data_vars:
        var_obj = regridder.source_grid_ds[v]
        if (
            var_obj.attrs.get("cf_role") == "mesh_topology"
            or "grid_mapping_name" in var_obj.attrs
        ):
            coords[v] = var_obj

    da_weights = xr.DataArray(
        row.reshape(regridder._shape_source),
        dims=regridder._dims_source,
        coords=coords,
        name="weights",
    )
    return da_weights


def plot_mesh(
    ds: "Union[xr.Dataset, str]",
    projection: Any = None,
    transform: Any = None,
    title: Optional[str] = None,
    edgecolor: str = "black",
    facecolor: str = "none",
    linewidth: float = 0.3,
    alpha: float = 1.0,
    ax: Any = None,
    figsize: tuple = (12, 8),
    **kwargs: Any,
) -> Any:
    """
    Plot the wireframe of an unstructured mesh (MPAS, UGRID, SCRIP, VTK).

    Draws each cell as a polygon outline on a Cartopy-projected map,
    producing images similar to the classic MPAS variable-resolution
    mesh visualizations.

    Parameters
    ----------
    ds : xr.Dataset or str
        Dataset containing unstructured mesh information (MPAS, UGRID,
        or SCRIP conventions), or a path to a VTK legacy file
        (``DATASET UNSTRUCTURED_GRID``) as written by ESMF.
    projection : cartopy.crs.Projection, optional
        Map projection for the axes. Defaults to ``ccrs.Orthographic``
        (globe view) if cartopy is available.
    transform : cartopy.crs.Projection, optional
        Coordinate reference for the vertex data. Defaults to
        ``ccrs.PlateCarree()``.
    title : str, optional
        Plot title.
    edgecolor : str, default "black"
        Color of cell edges.
    facecolor : str, default "none"
        Fill color of cells. Use ``"none"`` for wireframe.
    linewidth : float, default 0.3
        Width of cell edges.
    alpha : float, default 1.0
        Transparency of the mesh polygons.
    ax : matplotlib.axes.Axes, optional
        Pre-existing GeoAxes to draw on. A new figure is created if
        ``None``.
    figsize : tuple, default (12, 8)
        Figure size when creating a new figure.
    **kwargs : Any
        Additional keyword arguments passed to
        ``matplotlib.collections.PolyCollection``.

    Returns
    -------
    matplotlib.collections.PolyCollection
        The collection of cell polygons added to the axes.

    Raises
    ------
    ImportError
        If matplotlib or cartopy is not installed.
    ValueError
        If the dataset does not contain recognizable mesh connectivity.

    Examples
    --------
    >>> import xarray as xr
    >>> from xregrid.viz import plot_mesh
    >>> ds = xr.open_dataset("mpas_mesh.nc")
    >>> plot_mesh(ds, title="MPAS Variable-Resolution Mesh")

    >>> # From an ESMF VTK file
    >>> plot_mesh("esmf_mesh.vtk", title="ESMF Mesh")
    """
    if plt is None:
        raise ImportError(
            "Matplotlib is required for plot_mesh. "
            "Install it with `pip install matplotlib`."
        )
    if ccrs is None:
        raise ImportError(
            "Cartopy is required for plot_mesh. "
            "Install it with `pip install cartopy`."
        )

    import numpy as np
    from matplotlib.collections import PolyCollection

    # --- Extract cell polygons from the mesh connectivity ---
    if isinstance(ds, str):
        polygons = _read_vtk_polygons(ds)
    else:
        polygons = _extract_cell_polygons(ds)

    # --- Set up map projection ---
    if transform is None:
        transform = ccrs.PlateCarree()
    if projection is None:
        projection = ccrs.Orthographic(central_longitude=0, central_latitude=0)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=projection)

    # Use the axes' actual projection for coordinate transforms
    ax_projection = ax.projection if hasattr(ax, "projection") else projection

    # Transform vertices into the target projection
    projected_polys = []
    for poly in polygons:
        pts = ax_projection.transform_points(transform, poly[:, 0], poly[:, 1])
        # Filter out polygons that contain non-finite projected coords
        if np.all(np.isfinite(pts[:, :2])):
            projected_polys.append(pts[:, :2])

    collection = PolyCollection(
        projected_polys,
        edgecolors=edgecolor,
        facecolors=facecolor,
        linewidths=linewidth,
        alpha=alpha,
        transform=ax.transData,
        **kwargs,
    )
    ax.add_collection(collection)

    ax.set_global()
    ax.coastlines(linewidth=0.5, color="gray")

    if title is None:
        title = "Unstructured Mesh"
    ax.set_title(title)

    return collection


def _extract_cell_polygons(ds: xr.Dataset) -> list:
    """
    Build a list of cell polygons from an unstructured mesh dataset.

    Each polygon is an (N, 2) ndarray of ``[lon, lat]`` vertices in
    degrees. Supports MPAS, UGRID, and SCRIP conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with mesh connectivity information.

    Returns
    -------
    list of np.ndarray
        Each element is an (N, 2) array of vertex coordinates for one
        cell polygon.

    Raises
    ------
    ValueError
        If no supported mesh convention is detected.
    """
    import numpy as np
    from xregrid.grid import (
        _to_degrees,
        _clip_latitudes,
        _normalize_longitudes,
        _get_non_spatial_dims,
    )

    non_spatial_dims = _get_non_spatial_dims(ds)

    # --- MPAS ---
    if "verticesOnCell" in ds and "latVertex" in ds and "lonVertex" in ds:
        v_lat = ds["latVertex"]
        v_lon = ds["lonVertex"]
        v_conn = ds["verticesOnCell"]

        for name, da in [("lat", v_lat), ("lon", v_lon), ("conn", v_conn)]:
            isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
            if isel_dict:
                if name == "lat":
                    v_lat = v_lat.isel(isel_dict, drop=True)
                elif name == "lon":
                    v_lon = v_lon.isel(isel_dict, drop=True)
                else:
                    v_conn = v_conn.isel(isel_dict, drop=True)

        node_lat = _clip_latitudes(_to_degrees(v_lat)).values
        node_lon = _normalize_longitudes(_to_degrees(v_lon)).values
        conn_raw = v_conn.values  # 1-based
        n_edges = (
            ds["nEdgesOnCell"].values
            if "nEdgesOnCell" in ds
            else np.full(conn_raw.shape[0], conn_raw.shape[1])
        )

        polygons = []
        for i in range(conn_raw.shape[0]):
            ne = int(n_edges[i])
            vidx = conn_raw[i, :ne] - 1  # to 0-based
            lons = node_lon[vidx]
            lats = node_lat[vidx]
            # Handle dateline wrapping: if span > 180°, shift
            if lons.max() - lons.min() > 180:
                lons = np.where(lons < 180, lons + 360, lons)
            polygons.append(np.column_stack([lons, lats]))
        return polygons

    # --- UGRID ---
    conn_var = None
    mesh_var = None
    for var in ds.variables:
        if ds[var].attrs.get("cf_role") == "mesh_topology":
            mesh_var = var
            break
    if mesh_var:
        conn_var = ds[mesh_var].attrs.get("face_node_connectivity")
    if not conn_var:
        for var in ds.variables:
            if ds[var].attrs.get("cf_role") == "face_node_connectivity":
                conn_var = var
                break
    if not conn_var and "face_node_connectivity" in ds:
        conn_var = "face_node_connectivity"

    if conn_var:
        # Discover node coordinate variables
        node_lon_var = node_lat_var = None
        if mesh_var and mesh_var in ds:
            nc_attr = ds[mesh_var].attrs.get("node_coordinates", "").split()
            if len(nc_attr) >= 2:
                node_lon_var, node_lat_var = nc_attr[0], nc_attr[1]
        if not node_lon_var:
            for v in ds.variables:
                if ds[v].attrs.get("standard_name") == "longitude":
                    node_lon_var = v
                if ds[v].attrs.get("standard_name") == "latitude":
                    node_lat_var = v
        if not node_lon_var:
            node_lon_var = "node_lon" if "node_lon" in ds else None
            node_lat_var = "node_lat" if "node_lat" in ds else None

        if node_lon_var and node_lat_var:
            v_lon = ds[node_lon_var]
            v_lat = ds[node_lat_var]
            v_conn = ds[conn_var]
            for name, da in [("lon", v_lon), ("lat", v_lat), ("conn", v_conn)]:
                isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
                if isel_dict:
                    if name == "lon":
                        v_lon = v_lon.isel(isel_dict, drop=True)
                    elif name == "lat":
                        v_lat = v_lat.isel(isel_dict, drop=True)
                    else:
                        v_conn = v_conn.isel(isel_dict, drop=True)

            node_lon = _normalize_longitudes(_to_degrees(v_lon)).values
            node_lat = _clip_latitudes(_to_degrees(v_lat)).values
            conn_raw = v_conn.values
            start_index = int(ds[conn_var].attrs.get("start_index", 0))
            fill_value = ds[conn_var].attrs.get("_FillValue", -1)

            polygons = []
            for i in range(conn_raw.shape[0]):
                row = conn_raw[i]
                valid = row[row != fill_value] - start_index
                if len(valid) < 3:
                    continue
                lons = node_lon[valid]
                lats = node_lat[valid]
                if lons.max() - lons.min() > 180:
                    lons = np.where(lons < 180, lons + 360, lons)
                polygons.append(np.column_stack([lons, lats]))
            return polygons

    # --- SCRIP ---
    if "lat_b" in ds and "lon_b" in ds and ds["lat_b"].ndim == 2:
        v_lat_b = ds["lat_b"]
        v_lon_b = ds["lon_b"]
        for name, da in [("lat", v_lat_b), ("lon", v_lon_b)]:
            isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
            if isel_dict:
                if name == "lat":
                    v_lat_b = v_lat_b.isel(isel_dict, drop=True)
                else:
                    v_lon_b = v_lon_b.isel(isel_dict, drop=True)

        lat_b = _clip_latitudes(_to_degrees(v_lat_b)).values
        lon_b = _normalize_longitudes(_to_degrees(v_lon_b)).values

        polygons = []
        for i in range(lat_b.shape[0]):
            lons = lon_b[i]
            lats = lat_b[i]
            if lons.max() - lons.min() > 180:
                lons = np.where(lons < 180, lons + 360, lons)
            polygons.append(np.column_stack([lons, lats]))
        return polygons

    raise ValueError(
        "Could not detect unstructured mesh connectivity in the dataset. "
        "Supported conventions: MPAS (verticesOnCell), UGRID "
        "(face_node_connectivity), SCRIP (lat_b/lon_b)."
    )


def _read_vtk_polygons(path: str) -> list:
    """
    Parse a VTK legacy unstructured grid file into cell polygons.

    Reads the ASCII legacy VTK format (``DATASET UNSTRUCTURED_GRID``)
    as written by ESMF's ``Mesh._write_`` method. Points are
    interpreted as ``(lon, lat, [z])`` in degrees.

    Parameters
    ----------
    path : str
        Path to a ``.vtk`` file.

    Returns
    -------
    list of np.ndarray
        Each element is an ``(N, 2)`` array of ``[lon, lat]`` vertex
        coordinates for one cell polygon.

    Raises
    ------
    ValueError
        If the file is not a valid VTK legacy unstructured grid.
    """
    import numpy as np

    with open(path, "r") as f:
        lines = f.readlines()

    # --- Parse POINTS ---
    points = None
    cells_raw = []
    cell_types = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("POINTS"):
            parts = line.split()
            n_points = int(parts[1])
            coords = []
            i += 1
            while len(coords) < n_points * 3:
                coords.extend(lines[i].strip().split())
                i += 1
            points = np.array(coords, dtype=np.float64).reshape(n_points, 3)
            continue

        if line.startswith("CELLS"):
            parts = line.split()
            n_cells = int(parts[1])
            i += 1
            for _ in range(n_cells):
                vals = list(map(int, lines[i].strip().split()))
                n_verts = vals[0]
                cells_raw.append(vals[1 : 1 + n_verts])
                i += 1
            continue

        if line.startswith("CELL_TYPES"):
            n_types = int(line.split()[1])
            i += 1
            for _ in range(n_types):
                cell_types.append(int(lines[i].strip()))
                i += 1
            continue

        i += 1

    if points is None or len(cells_raw) == 0:
        raise ValueError(
            f"Could not parse VTK file '{path}'. Expected a legacy "
            "UNSTRUCTURED_GRID with POINTS and CELLS sections."
        )

    # VTK cell types for 2D polygonal cells:
    # 5 = VTK_TRIANGLE, 7 = VTK_POLYGON, 9 = VTK_QUAD
    poly_types = {5, 7, 9}

    # Points are (lon, lat, z) — take first two columns
    lons = points[:, 0]
    lats = points[:, 1]

    polygons = []
    for idx, cell_verts in enumerate(cells_raw):
        # Skip non-polygonal cells if cell_types are present
        if cell_types and cell_types[idx] not in poly_types:
            continue
        vidx = np.array(cell_verts)
        cell_lons = lons[vidx].copy()
        cell_lats = lats[vidx].copy()
        if cell_lons.max() - cell_lons.min() > 180:
            cell_lons = np.where(cell_lons < 180, cell_lons + 360, cell_lons)
        polygons.append(np.column_stack([cell_lons, cell_lats]))

    return polygons
