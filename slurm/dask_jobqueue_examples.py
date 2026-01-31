"""
Examples of using dask-jobqueue on NOAA RDHPCS machines.

These examples show how to set up a Dask cluster using the `dask-jobqueue` library
on Hera, Jet, Gaea, and Ursa. This allows for seamless scaling of XRegrid operations
across multiple compute nodes.

Prerequisites:
    pip install dask-jobqueue
"""

import xarray as xr
from dask_jobqueue import SLURMCluster
from distributed import Client
from xregrid import Regridder, create_global_grid


def hera_example():
    """Example configuration for the Hera system."""
    cluster = SLURMCluster(
        queue="hera",
        account="your_project_account",  # e.g., 'ovp'
        cores=40,
        processes=40,  # 1 process per core, or adjust as needed
        memory="160GB",
        walltime="01:00:00",
        job_extra_directives=["--nodes=1", "--exclusive"],
        local_directory="/scratch2/NCEPDEV/stmp1/Your.Name/dask-local",
    )
    return cluster


def jet_example():
    """Example configuration for the Jet system."""
    # Note: Jet has multiple partitions (sjet, vjet, xjet, kjet).
    # If no partition is specified, SLURM will assign one.
    cluster = SLURMCluster(
        queue="batch",
        account="your_project_account",
        cores=20,
        processes=10,  # 2 cores per process
        memory="120GB",
        walltime="00:30:00",
        job_extra_directives=["--qos=batch"],
        local_directory="/lfs4/HFIP/hfvip/Your.Name/dask-local",
    )
    return cluster


def gaea_example(cluster_name="c5"):
    """Example configuration for the Gaea system."""
    # Gaea requires specifying the cluster (-M c5 or -M c6)
    cores = 128 if cluster_name == "c5" else 192
    memory = "256GB" if cluster_name == "c5" else "384GB"

    cluster = SLURMCluster(
        queue="batch",
        account="your_project_account",
        cores=cores,
        processes=16,  # 8 or 12 cores per process
        memory=memory,
        walltime="01:00:00",
        job_extra_directives=[f"-M {cluster_name}"],
        local_directory="/lustre/f2/dev/Your.Name/dask-local",
    )
    return cluster


def ursa_example():
    """Example configuration for the Ursa system."""
    cluster = SLURMCluster(
        queue="u1-compute",
        account="your_project_account",
        cores=192,
        processes=32,  # 6 cores per process
        memory="384GB",
        walltime="01:00:00",
        job_extra_directives=["--exclusive"],
        local_directory="/scratch3/NCEPDEV/stmp1/Your.Name/dask-local",
    )
    return cluster


def run_regridding(cluster):
    """Generic function to run regridding on a cluster."""
    # Scale the cluster
    cluster.scale(jobs=2)  # Request 2 nodes

    # Connect to the cluster
    client = Client(cluster)
    print(f"Cluster Dashboard: {client.dashboard_link}")

    # 1. Create large synthetic data
    ds = xr.Dataset(
        {
            "air": (
                ("time", "lat", "lon"),
                xr.tutorial.open_dataset("air_temperature").air.data,
            )
        }
    ).chunk({"time": 1, "lat": -1, "lon": -1})

    # 2. Define target grid
    target_grid = create_global_grid(res_lat=0.5, res_lon=0.5)

    # 3. Initialize Regridder with parallel=True
    # This will use the Dask cluster to generate weights in parallel
    regridder = Regridder(
        ds, target_grid, method="bilinear", periodic=True, parallel=True
    )

    # 4. Apply regridding
    # The application itself will also be parallelized across the cluster
    air_regridded = regridder(ds.air)

    # 5. Compute the result
    result = air_regridded.compute()
    print("Regridding complete.")
    return result


if __name__ == "__main__":
    # Choose your system
    # cluster = hera_example()
    # cluster = ursa_example()
    # run_regridding(cluster)
    print("Check the script for system-specific examples.")
