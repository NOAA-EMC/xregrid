"""
Create Sample Data for xregrid CLI
=================================

This script generates a sample global NetCDF file that can be used to test
the xregrid CLI. It creates a 2-degree global grid with a synthetic
sine-cosine wave pattern.
"""

import numpy as np
from xregrid.utils import create_global_grid


def create_sample_data():
    # Create a 2-degree global grid
    ds = create_global_grid(2.0, 2.0)

    # Add some dummy data
    lat = ds.lat.values
    lon = ds.lon.values
    data = (
        np.sin(np.deg2rad(lat))[:, np.newaxis] * np.cos(np.deg2rad(lon))[np.newaxis, :]
    )

    ds["sample_var"] = (["lat", "lon"], data)
    ds["sample_var"].attrs["units"] = "dimensionless"

    ds.to_netcdf("sample_input.nc")
    print("Created sample_input.nc")


if __name__ == "__main__":
    create_sample_data()
