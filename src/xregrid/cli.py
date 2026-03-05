from __future__ import annotations

import argparse
import os
import sys

import xarray as xr
from xregrid import Regridder, create_global_grid, create_regional_grid
from xregrid.utils import get_rdhpcs_cluster


def parse_args():
    parser = argparse.ArgumentParser(description="xregrid CLI: Regrid NetCDF files.")
    parser.add_argument("src", help="Path to the source NetCDF file.")
    parser.add_argument(
        "target",
        help="Path to the target NetCDF file or resolution in degrees (e.g., 1.0).",
    )
    parser.add_argument(
        "--method",
        default="bilinear",
        choices=["bilinear", "conservative", "nearest_s2d", "nearest_d2s", "patch"],
        help="Regridding method (default: bilinear).",
    )
    parser.add_argument(
        "--output", "-o", default="output.nc", help="Path to the output NetCDF file."
    )
    parser.add_argument(
        "--extent",
        help="Target grid extent as min_lat,max_lat,min_lon,max_lon (only used if target is a resolution).",
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Set to True for global grids with periodic boundaries.",
    )
    parser.add_argument(
        "--reuse-weights",
        action="store_true",
        help="Reuse weights if the weights file already exists.",
    )
    parser.add_argument(
        "--weights-file", default="weights.nc", help="Path to the weights file."
    )
    parser.add_argument(
        "--skipna", action="store_true", help="Handle NaNs by re-normalizing weights."
    )

    # Dask options
    dask_group = parser.add_argument_group("Dask options")
    dask_group.add_argument(
        "--dask-local",
        type=int,
        metavar="N_WORKERS",
        help="Start a local Dask cluster with N_WORKERS.",
    )
    dask_group.add_argument(
        "--dask-scheduler",
        metavar="ADDRESS",
        help="Connect to an existing Dask scheduler at ADDRESS.",
    )
    dask_group.add_argument(
        "--dask-jobqueue",
        metavar="MACHINE",
        help="Use dask-jobqueue SLURMCluster for MACHINE (e.g., hera, jet).",
    )
    dask_group.add_argument("--dask-account", help="SLURM account for dask-jobqueue.")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup Dask Client if requested
    client = None
    if args.dask_local:
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=args.dask_local)
        client = Client(cluster)
        print(f"Started local Dask cluster: {client.scheduler_address}")
    elif args.dask_scheduler:
        from dask.distributed import Client

        client = Client(args.dask_scheduler)
        print(f"Connected to Dask scheduler: {args.dask_scheduler}")
    elif args.dask_jobqueue:
        cluster = get_rdhpcs_cluster(
            machine=args.dask_jobqueue, account=args.dask_account
        )
        from dask.distributed import Client

        client = Client(cluster)
        print(f"Started dask-jobqueue cluster on {args.dask_jobqueue}")

    try:
        # 2. Load Source Dataset
        print(f"Loading source dataset: {args.src}")
        ds_src = xr.open_dataset(args.src, chunks={})

        # 3. Load or Create Target Grid
        if os.path.exists(args.target):
            print(f"Loading target grid: {args.target}")
            ds_tgt = xr.open_dataset(args.target, chunks={})
        else:
            try:
                res = float(args.target)
                if args.extent:
                    lat_min, lat_max, lon_min, lon_max = map(
                        float, args.extent.split(",")
                    )
                    print(
                        f"Creating regional target grid: res={res}, extent=[{lat_min}, {lat_max}, {lon_min}, {lon_max}]"
                    )
                    ds_tgt = create_regional_grid(
                        (lat_min, lat_max), (lon_min, lon_max), res, res
                    )
                else:
                    print(f"Creating global target grid: res={res}")
                    ds_tgt = create_global_grid(res, res)
            except ValueError:
                print(
                    f"Error: target '{args.target}' is neither a file nor a valid resolution."
                )
                sys.exit(1)

        # 4. Initialize Regridder
        print(
            f"Initializing Regridder (method={args.method}, periodic={args.periodic})"
        )
        regridder = Regridder(
            ds_src,
            ds_tgt,
            method=args.method,
            periodic=args.periodic,
            reuse_weights=args.reuse_weights,
            filename=args.weights_file,
            skipna=args.skipna,
            parallel=(client is not None),
        )

        # 5. Perform Regridding
        print("Regridding dataset...")
        ds_out = regridder(ds_src)

        # 6. Save Output
        print(f"Saving output to: {args.output}")
        ds_out.to_netcdf(args.output)
        print("Done.")

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
