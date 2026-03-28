from __future__ import annotations

import numpy as np
import xarray as xr

from xregrid.regridder import Regridder
from xregrid.utils import create_global_grid


def test_regridder_refactor_consistency():
    """
    Double-Check Test: Verify Regridder still works with refactored constants.
    Ensures NumPy and Dask backends produce identical results.
    """
    # 1. Setup small grids
    res = 10.0
    ds_src = create_global_grid(res, res)
    ds_tgt = create_global_grid(res * 2, res * 2)

    # Add some data
    data = np.random.rand(*ds_src.lat.shape, *ds_src.lon.shape)
    da_src = xr.DataArray(
        data,
        coords={c: ds_src.coords[c] for c in ["lat", "lon"]},
        dims=("lat", "lon"),
        name="test_data",
    )

    # 2. Eager execution (NumPy)
    regridder_eager = Regridder(ds_src, ds_tgt, method="bilinear")
    res_eager = regridder_eager(da_src)

    # 3. Lazy execution (Dask)
    da_lazy = da_src.chunk({"lat": 5, "lon": 5})
    # We can reuse the same regridder as it's backend-agnostic for application
    res_lazy = regridder_eager(da_lazy)

    # 4. Assertions
    # Verify results are identical
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Verify regridder attributes still match expectations
    assert regridder_eager.method == "bilinear"
    assert "Regridded using xregrid.Regridder" in res_eager.attrs["history"]

    # Check that constants were correctly applied (mocked or real)
    try:
        import esmpy

        expected_method = esmpy.RegridMethod.BILINEAR
        assert regridder_eager.method_map["bilinear"] == expected_method
    except ImportError:
        pass


def test_regridder_extrap_refactor():
    """Verify extrapolation method refactor."""
    res = 10.0
    ds_src = create_global_grid(res, res)
    ds_tgt = create_global_grid(res * 2, res * 2)

    regridder = Regridder(
        ds_src, ds_tgt, method="nearest_s2d", extrap_method="nearest_idw"
    )

    assert regridder.extrap_method == "nearest_idw"

    try:
        import esmpy

        expected_extrap = esmpy.ExtrapMethod.NEAREST_IDAVG
        assert regridder.extrap_method_map["nearest_idw"] == expected_extrap
    except ImportError:
        pass
