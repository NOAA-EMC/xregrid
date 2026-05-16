from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xregrid import Regridder, create_global_grid


def test_dataarray_accessor_to_regridder():
    """
    Verify DataArray accessor works with a Regridder instance (Eager & Lazy).

    Follows the Aero Protocol 'Double-Check' Rule.
    """
    ds_src = create_global_grid(10, 10)
    ds_tgt = create_global_grid(5, 5)

    # 1. Eager (NumPy)
    da_src = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
        name="test_da",
    )

    # Create regridder once
    regridder = da_src.regrid.get_regridder(ds_tgt)
    assert isinstance(regridder, Regridder)

    # Regrid using target_grid (Dataset)
    res_with_ds = da_src.regrid.to(ds_tgt)

    # Regrid using regridder instance (New functionality)
    res_with_regridder = da_src.regrid.to(regridder)

    # Verification
    xr.testing.assert_allclose(res_with_ds, res_with_regridder)

    # 2. Lazy (Dask)
    if da is not None:
        da_lazy = da_src.chunk({"lat": 9, "lon": 9})
        res_lazy = da_lazy.regrid.to(regridder)

        assert hasattr(res_lazy.data, "dask")
        # Numerical Verification
        xr.testing.assert_allclose(res_with_ds, res_lazy.compute())


def test_dataset_accessor_to_regridder():
    """
    Verify Dataset accessor works with a Regridder instance.
    """
    ds_src = create_global_grid(10, 10)
    ds_tgt = create_global_grid(5, 5)

    ds = xr.Dataset(
        {"var1": (("lat", "lon"), np.random.rand(18, 36))},
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
    )

    # Create regridder once
    regridder = ds.regrid.get_regridder(ds_tgt)
    assert isinstance(regridder, Regridder)

    # Regrid using target_grid (Dataset)
    res_with_ds = ds.regrid.to(ds_tgt)

    # Regrid using regridder instance
    res_with_regridder = ds.regrid.to(regridder)

    # Verification
    xr.testing.assert_allclose(res_with_ds, res_with_regridder)


def test_accessor_plot_diagnostics_smoke():
    """
    Smoke test for accessor plotting methods.
    """
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    ds_src = create_global_grid(30, 30)
    ds_tgt = create_global_grid(10, 10)
    da = xr.DataArray(
        np.random.rand(6, 12),
        dims=("lat", "lon"),
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
    )

    # Smoke test: simply check if it runs without error
    fig = da.regrid.plot_diagnostics(ds_tgt)
    assert fig is not None
    plt.close(fig)

    ds = da.to_dataset(name="test")
    fig2 = ds.regrid.plot_diagnostics(ds_tgt)
    assert fig2 is not None
    plt.close(fig2)
