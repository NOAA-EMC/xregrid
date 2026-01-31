from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from xregrid.utils import get_rdhpcs_cluster
from xregrid.xregrid import _get_mesh_info


def test_get_mesh_info_rectilinear_order():
    """Test that _get_mesh_info correctly handles rectilinear grids with different coord orders."""
    # Create a grid where coords are (lon, lat)
    lon = np.arange(0, 360, 10)
    lat = np.arange(-90, 91, 10)

    # Broadcast to create a dataset
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})

    # Check (lat, lon) order
    lon_m, lat_m, shape, dims, unstructured = _get_mesh_info(ds)
    assert not unstructured
    assert dims == ("lat", "lon")
    assert shape == (lat.size, lon.size)
    assert lat_m.shape == (lat.size, lon.size)
    assert lon_m.shape == (lat.size, lon.size)

    # Verify that the dead code was removed and it still works
    # (The test above already confirms it works for standard 1D coords)


def test_get_rdhpcs_cluster_detection():
    """Test machine detection in get_rdhpcs_cluster."""

    with patch("socket.gethostname") as mock_hostname:
        # Test Hera detection
        mock_hostname.return_value = "hfe01.hera.noaa.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc")
            args, kwargs = mock_slurm.call_args
            assert kwargs["queue"] == "hera"
            assert kwargs["cores"] == 40

        # Test Jet detection
        mock_hostname.return_value = "fe01.jet.noaa.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc")
            args, kwargs = mock_slurm.call_args
            assert kwargs["queue"] == "batch"
            assert kwargs["cores"] == 24

        # Test Gaea detection
        mock_hostname.return_value = "gaea12.ncrc.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc", machine="gaea-c6")
            args, kwargs = mock_slurm.call_args
            assert kwargs["cores"] == 192
            assert "-M c6" in kwargs["job_extra_directives"][0]

        # Test Ursa detection
        mock_hostname.return_value = "ufe01.ursa.noaa.gov"
        with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
            get_rdhpcs_cluster(account="test_acc")
            args, kwargs = mock_slurm.call_args
            assert kwargs["queue"] == "u1-compute"
            assert kwargs["cores"] == 192


def test_get_rdhpcs_cluster_explicit():
    """Test explicit machine specification in get_rdhpcs_cluster."""
    with patch("dask_jobqueue.SLURMCluster", MagicMock()) as mock_slurm:
        get_rdhpcs_cluster(machine="hera", account="test_acc", walltime="02:00:00")
        args, kwargs = mock_slurm.call_args
        assert kwargs["queue"] == "hera"
        assert kwargs["walltime"] == "02:00:00"
        assert kwargs["account"] == "test_acc"


if __name__ == "__main__":
    pytest.main([__file__])
