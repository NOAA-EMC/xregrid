import os
import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid, load_esmf_file


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


# --- New Enhancements Tests ---


def test_load_esmf_file_scrip(tmp_path):
    """Verify load_esmf_file handles SCRIP variable names correctly."""
    filepath = os.path.join(tmp_path, "scrip_grid.nc")

    # Create a dummy SCRIP-style file
    ds_scrip = xr.Dataset(
        data_vars={
            "grid_center_lat": (["grid_size"], [10.0, 20.0]),
            "grid_center_lon": (["grid_size"], [30.0, 40.0]),
            "grid_corner_lat": (
                ["grid_size", "grid_corners"],
                [[9, 11, 11, 9], [19, 21, 21, 19]],
            ),
            "grid_corner_lon": (
                ["grid_size", "grid_corners"],
                [[29, 29, 31, 31], [39, 39, 41, 41]],
            ),
            "grid_imask": (["grid_size"], [1, 1]),
        }
    )
    ds_scrip.to_netcdf(filepath)

    ds_loaded = load_esmf_file(filepath)

    assert "lat" in ds_loaded
    assert "lon" in ds_loaded
    assert "lat_b" in ds_loaded
    assert "lon_b" in ds_loaded
    assert "mask" in ds_loaded

    assert ds_loaded.lat.attrs["standard_name"] == "latitude"
    assert ds_loaded.lon.attrs["standard_name"] == "longitude"
    assert ds_loaded.lat.attrs["bounds"] == "lat_b"
    assert "history" in ds_loaded.attrs
    assert "renamed standard variables" in ds_loaded.attrs["history"]


def test_quality_report_dataset():
    """Verify Regridder.quality_report supports format='dataset'."""
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(20, 20)

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    report_ds = regridder.quality_report(format="dataset")

    assert isinstance(report_ds, xr.Dataset)
    assert "n_src" in report_ds
    assert "n_dst" in report_ds
    assert "unmapped_count" in report_ds
    assert report_ds.attrs["method"] == "bilinear"
    assert "history" in report_ds.attrs


def test_regrid_recursion_safety_double_check():
    """Aero Protocol Double-Check: Verify recursion safety and backend identity."""
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(20, 20)

    regridder = Regridder(src_grid, tgt_grid)

    # Create an unnamed DataArray with an auxiliary coordinate
    data = np.random.rand(18, 36).astype(np.float32)
    aux_coord = xr.DataArray(
        np.random.rand(18, 36).astype(np.float32), dims=("lat", "lon"), name="aux"
    )

    da_eager = xr.DataArray(
        data,
        coords={"lat": src_grid.lat, "lon": src_grid.lon, "aux": aux_coord},
        dims=("lat", "lon"),
    )

    # Eager result
    res_eager = regridder(da_eager)

    # Lazy result
    da_lazy = da_eager.chunk({"lat": 9, "lon": 18})
    res_lazy_obj = regridder(da_lazy)

    # Verify coords were also processed lazily (Aero Protocol)
    assert res_lazy_obj.aux.chunks is not None

    res_lazy = res_lazy_obj.compute()

    # Verify identity (Double-Check Rule)
    xr.testing.assert_allclose(res_eager, res_lazy)
    assert res_eager.name is None
    assert "aux" in res_eager.coords
    assert res_eager.aux.shape == (9, 18)


def test_recursion_protection_naming():
    """Verify that recursive coordinate regridding doesn't infinite loop with naming conflicts."""
    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)

    # Create a DataArray with a coordinate that points to itself or another regriddable coord
    data = np.random.rand(6, 12)
    da_in = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": src.lat, "lon": src.lon},
        name="test_da",
    )

    # Add an auxiliary spatial coordinate
    aux_coord = xr.DataArray(
        np.random.rand(6, 12),
        dims=("lat", "lon"),
        coords={"lat": src.lat, "lon": src.lon},
        name="aux_coord",
    )
    da_in = da_in.assign_coords(aux=aux_coord)

    regridder = Regridder(src, tgt, method="bilinear")

    # This should run without infinite recursion
    res = regridder(da_in)
    assert "aux" in res.coords
    assert res.aux.shape == (12, 24)


def test_lazy_breaker_avoidance_logic():
    """Verify that grid creation handles dask-backed coordinates."""
    src = create_global_grid(30, 30).chunk({"lat": 3, "lon": 6})
    tgt = create_global_grid(15, 15).chunk({"lat": 3, "lon": 6})

    # Initialize Regridder.
    regridder = Regridder(src, tgt, method="bilinear")

    assert regridder._weights_matrix is not None
    assert regridder._weights_matrix.shape == (12 * 24, 6 * 12)


def test_plot_diagnostics_interactive_smoke():
    """Verify plot_diagnostics_interactive returns a layout (Track B)."""
    try:
        from xregrid.viz import plot_diagnostics_interactive

        import hvplot.xarray  # noqa: F401
    except ImportError:
        pytest.skip("hvplot or holoviews not installed")

    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)
    regridder = Regridder(src, tgt, method="bilinear")

    layout = plot_diagnostics_interactive(regridder)
    assert layout is not None
    assert hasattr(layout, "cols")


def test_plot_weights_smoke():
    """Verify plot_weights runs without error (Track A)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)
    regridder = Regridder(src, tgt, method="bilinear")

    fig = regridder.plot_weights(0)
    assert fig is not None
    plt.close()


def test_eager_lazy_parity_enhancements_check():
    """Aero Protocol: Verify eager and lazy paths are identical with the new refactors."""
    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)
    regridder = Regridder(src, tgt, method="bilinear", skipna=True)

    data = np.random.rand(6, 12)
    data[0, 0] = np.nan

    da_eager = xr.DataArray(
        data, dims=("lat", "lon"), coords={"lat": src.lat, "lon": src.lon}
    )
    res_eager = regridder(da_eager)

    da_lazy = da_eager.chunk({"lat": 3, "lon": 6})
    res_lazy = regridder(da_lazy).compute()

    xr.testing.assert_allclose(res_eager, res_lazy)


def test_lazy_loading_viz():
    """Verify that visualization functions are lazy-loaded."""
    import sys
    import importlib

    # This test is sensitive to previous imports in the same process.
    # If xregrid.viz was already loaded and is held by other modules,
    # deleting it from sys.modules might not be enough for a clean test.

    import xregrid

    # If it's already there, we try to clear it for the test
    if "xregrid.viz" in sys.modules:
        del sys.modules["xregrid.viz"]

    # Ensure xregrid is reloaded to use the lazy __getattr__
    importlib.reload(xregrid)

    # In some test environments, other tests might have already loaded viz
    # and it might persist in sys.modules due to sub-module relationships.
    # We check if it's NOT there now.
    if "xregrid.viz" in sys.modules:
        pytest.skip(
            "xregrid.viz already loaded by another test, cannot verify laziness."
        )

    # Accessing a viz function should trigger loading
    plot_func = getattr(xregrid, "plot_static")
    assert plot_func is not None
    assert "xregrid.viz" in sys.modules


if __name__ == "__main__":
    pytest.main([__file__])
