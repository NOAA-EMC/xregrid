import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_quality_report_metrics():
    """Verify that quality_report returns expected keys and types."""
    src_res = 10
    tgt_res = 5
    src_grid = create_global_grid(src_res, src_res)
    tgt_grid = create_global_grid(tgt_res, tgt_res)

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    report = regridder.quality_report()

    assert isinstance(report, dict)
    expected_keys = {
        "unmapped_count",
        "unmapped_fraction",
        "weight_sum_min",
        "weight_sum_max",
        "weight_sum_mean",
        "n_src",
        "n_dst",
        "n_weights",
        "method",
        "periodic",
    }
    assert expected_keys.issubset(report.keys())
    assert isinstance(report["unmapped_count"], int)
    assert isinstance(report["unmapped_fraction"], float)
    assert report["n_src"] == 18 * 36
    assert report["n_dst"] == 36 * 72


def test_weights_to_xarray_export():
    """Verify that weights_to_xarray returns a valid xarray Dataset."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(10, 10)
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    ds_weights = regridder.weights_to_xarray()

    assert isinstance(ds_weights, xr.Dataset)
    assert "row" in ds_weights
    assert "col" in ds_weights
    assert "S" in ds_weights
    assert ds_weights.attrs["method"] == "bilinear"
    assert ds_weights.attrs["n_src"] == 18 * 36
    assert ds_weights.attrs["n_dst"] == 18 * 36


def test_repr_transparency():
    """Verify that __repr__ contains quality information."""
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(30, 30)
    regridder = Regridder(src_grid, tgt_grid)

    repr_str = repr(regridder)
    assert "Regridder" in repr_str
    assert "unmapped=" in repr_str


def test_aero_identity_with_diagnostics():
    """
    Aero Protocol: Verify that regridding results are identical for Eager and Lazy data,
    and that diagnostics remain consistent.
    """
    # Small grid for fast testing
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(15, 15)
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # 1. Eager Data
    data = np.random.rand(6, 12)
    da_eager = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
        name="test_data",
    )
    res_eager = regridder(da_eager)

    # 2. Lazy Data
    da_lazy = da_eager.chunk({"lat": 3, "lon": 6})
    res_lazy = regridder(da_lazy)

    # Verify identity (Eager vs Lazy)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Verify that the Regridder itself is unchanged and diagnostics work
    report = regridder.quality_report()
    assert report["n_src"] == 6 * 12
    assert report["n_dst"] == 12 * 24


if __name__ == "__main__":
    pytest.main([__file__])
