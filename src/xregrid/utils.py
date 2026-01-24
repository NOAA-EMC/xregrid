from __future__ import annotations

import datetime
from typing import Tuple

import numpy as np
import xarray as xr


def create_global_grid(
    res_lat: float,
    res_lon: float,
    add_bounds: bool = True,
) -> xr.Dataset:
    """
    Create a global rectilinear grid dataset.

    Parameters
    ----------
    res_lat : float
        Latitude resolution in degrees.
    res_lon : float
        Longitude resolution in degrees.
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.

    Returns
    -------
    xr.Dataset
        The global grid dataset containing 'lat' and 'lon'.
    """
    lat = np.arange(-90 + res_lat / 2, 90, res_lat)
    lon = np.arange(0 + res_lon / 2, 360, res_lon)

    ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    if add_bounds:
        lat_b = np.arange(-90, 90 + res_lat, res_lat)
        lon_b = np.arange(0, 360 + res_lon, res_lon)
        ds.coords["lat_b"] = (["lat_b"], lat_b, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["lon_b"], lon_b, {"units": "degrees_east"})

        # Link bounds using cf-xarray convention
        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["history"] = (
        f"{timestamp}: Created global grid ({res_lat}x{res_lon}) using xregrid."
    )

    return ds


def create_regional_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    res_lat: float,
    res_lon: float,
    add_bounds: bool = True,
) -> xr.Dataset:
    """
    Create a regional rectilinear grid dataset.

    Parameters
    ----------
    lat_range : tuple of float
        (min_lat, max_lat).
    lon_range : tuple of float
        (min_lon, max_lon).
    res_lat : float
        Latitude resolution in degrees.
    res_lon : float
        Longitude resolution in degrees.
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.

    Returns
    -------
    xr.Dataset
        The regional grid dataset containing 'lat' and 'lon'.
    """
    lat = np.arange(lat_range[0] + res_lat / 2, lat_range[1], res_lat)
    lon = np.arange(lon_range[0] + res_lon / 2, lon_range[1], res_lon)

    ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    if add_bounds:
        lat_b = np.arange(lat_range[0], lat_range[1] + res_lat, res_lat)
        lon_b = np.arange(lon_range[0], lon_range[1] + res_lon, res_lon)
        ds.coords["lat_b"] = (["lat_b"], lat_b, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["lon_b"], lon_b, {"units": "degrees_east"})

        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["history"] = (
        f"{timestamp}: Created regional grid ({res_lat}x{res_lon}) using xregrid."
    )

    return ds


def load_esmf_file(filepath: str) -> xr.Dataset:
    """
    Load an ESMF mesh, mosaic, or grid file into an xarray Dataset.

    Parameters
    ----------
    filepath : str
        Path to the ESMF file.

    Returns
    -------
    xr.Dataset
        The dataset representation of the ESMF file.
    """
    # This is a basic implementation using xarray to open NetCDF files
    # which many ESMF files are.
    ds = xr.open_dataset(filepath)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history = f"{timestamp}: Loaded ESMF file from {filepath}."
    if "history" in ds.attrs:
        ds.attrs["history"] = f"{history}\n" + ds.attrs["history"]
    else:
        ds.attrs["history"] = history

    return ds
