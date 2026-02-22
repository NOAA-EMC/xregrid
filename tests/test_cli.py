import subprocess
import sys
import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch


@pytest.fixture
def sample_input(tmp_path):
    path = tmp_path / "input.nc"
    lat = np.arange(-89, 90, 2)
    lon = np.arange(1, 360, 2)
    data = np.random.rand(len(lat), len(lon))
    ds = xr.Dataset(
        data_vars={"test": (["lat", "lon"], data)},
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path)
    return path


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "xregrid.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "xregrid CLI" in result.stdout


def test_cli_basic(sample_input, tmp_path, monkeypatch):
    output = tmp_path / "output.nc"
    # We need to mock Regridder because esmpy is not installed
    with patch("xregrid.cli.Regridder") as mock_regridder:
        # Mock the regridder instance and its __call__ method
        instance = mock_regridder.return_value
        instance.return_value = xr.open_dataset(
            sample_input
        )  # Return input as mock output

        # Mock sys.argv
        test_args = [
            "xregrid.cli",
            str(sample_input),
            "1.0",
            "--output",
            str(output),
            "--method",
            "bilinear",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        from xregrid.cli import main

        main()

        assert output.exists()
        mock_regridder.assert_called_once()
        args, kwargs = mock_regridder.call_args
        assert kwargs["method"] == "bilinear"
        assert args[1].lat.size == 180  # 1.0 degree global grid has 180 lat points


def test_cli_regional(sample_input, tmp_path, monkeypatch):
    output = tmp_path / "output.nc"
    with patch("xregrid.cli.Regridder") as mock_regridder:
        instance = mock_regridder.return_value
        instance.return_value = xr.open_dataset(sample_input)

        test_args = [
            "xregrid.cli",
            str(sample_input),
            "0.5",
            "--output",
            str(output),
            "--extent=-10,10,20,40",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        from xregrid.cli import main

        main()

        assert output.exists()
        mock_regridder.assert_called_once()
        target_grid = mock_regridder.call_args[0][1]
        assert target_grid.lat.min() >= -10
        assert target_grid.lat.max() <= 10
        assert target_grid.lon.min() >= 20
        assert target_grid.lon.max() <= 40
