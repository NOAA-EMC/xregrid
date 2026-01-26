import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_extrap_method_persistence():
    """Verify that extrap_method is correctly stored and persisted."""
    src = create_global_grid(30, 30)
    tgt = create_global_grid(10, 10)

    # 1. Test weight generation with extrap_method
    regridder = Regridder(
        src,
        tgt,
        extrap_method="nearest_idw",
        extrap_dist_exponent=3.0,
        reuse_weights=True,
        filename="test_extrap.nc",
    )
    assert regridder.extrap_method == "nearest_idw"
    assert regridder.extrap_dist_exponent == 3.0

    # 2. Test loading
    regridder_loaded = Regridder(
        src,
        tgt,
        extrap_method="nearest_idw",
        reuse_weights=True,
        filename="test_extrap.nc",
    )
    assert regridder_loaded.extrap_method == "nearest_idw"

    # 3. Test validation failure on mismatch
    with pytest.raises(ValueError, match="does not match"):
        Regridder(
            src,
            tgt,
            extrap_method="creep_fill",
            reuse_weights=True,
            filename="test_extrap.nc",
        )


def test_coordinate_preservation():
    """Verify that non-spatial coordinates are preserved in Dataset regridding."""
    src = create_global_grid(30, 30)
    tgt = create_global_grid(10, 10)

    ds_in = xr.Dataset(
        {"temp": (["lat", "lon"], np.random.rand(6, 12))},
        coords={
            "lat": src.lat,
            "lon": src.lon,
            "scalar_coord": 42,
            "time": ("time", [np.datetime64("2020-01-01")]),
        },
    )

    regridder = Regridder(src, tgt)
    ds_out = regridder(ds_in)

    assert "scalar_coord" in ds_out.coords
    assert ds_out.coords["scalar_coord"] == 42
    assert "time" in ds_out.coords
    assert ds_out.time.values[0] == np.datetime64("2020-01-01")


def test_plot_static_nd_warning():
    """Verify that plot_static handles N-D arrays with a warning."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    da = xr.DataArray(
        np.random.rand(5, 18, 36),
        dims=("time", "lat", "lon"),
        coords={"lat": np.linspace(-90, 90, 18), "lon": np.linspace(0, 360, 36)},
    )

    from xregrid import plot_static

    with pytest.warns(UserWarning, match="DataArray has 3 dimensions"):
        plot_static(da)
    plt.close("all")


def test_eager_lazy_identity_extrap():
    """Aero Protocol Double-Check: Verify eager and lazy results are identical."""
    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)

    regridder = Regridder(src, tgt, extrap_method="nearest_s2d")

    data = np.random.rand(6, 12)
    da_eager = xr.DataArray(
        data, dims=("lat", "lon"), coords={"lat": src.lat, "lon": src.lon}
    )
    res_eager = regridder(da_eager)

    da_lazy = da_eager.chunk({"lat": 3, "lon": 6})
    res_lazy = regridder(da_lazy).compute()

    xr.testing.assert_allclose(res_eager, res_lazy)


if __name__ == "__main__":
    pytest.main([__file__])
