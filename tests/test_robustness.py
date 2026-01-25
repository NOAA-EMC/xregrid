import os
import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_eager_lazy_identity_and_name_preservation():
    """Verify Eager and Lazy results are identical and name/history are preserved."""
    ds_in = create_global_grid(10, 10)
    ds_out = create_global_grid(5, 5)
    regridder = Regridder(ds_in, ds_out, method="bilinear")

    data = np.random.rand(18, 36)
    coords = {"lat": ds_in.lat, "lon": ds_in.lon}
    name = "test_var"
    attrs = {"units": "K", "history": "original history"}

    # 1. Eager (NumPy)
    da_eager = xr.DataArray(
        data, dims=("lat", "lon"), coords=coords, name=name, attrs=attrs
    )
    res_eager = regridder(da_eager)

    # 2. Lazy (Dask)
    da_lazy = xr.DataArray(
        data, dims=("lat", "lon"), coords=coords, name=name, attrs=attrs
    ).chunk({"lat": 9, "lon": 18})
    res_lazy = regridder(da_lazy)

    # Identity check
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Metadata checks
    for res in [res_eager, res_lazy]:
        assert res.name == name
        assert res.attrs["units"] == "K"
        assert "original history" in res.attrs["history"]
        assert "Regridded" in res.attrs["history"]


def test_weight_validation(tmp_path):
    """Verify that loaded weights are validated against the grid."""
    ds_in = create_global_grid(10, 10)
    ds_out = create_global_grid(5, 5)

    weights_file = str(tmp_path / "weights.nc")

    # Generate weights for 10x10 -> 5x5
    Regridder(ds_in, ds_out, reuse_weights=True, filename=weights_file)
    assert os.path.exists(weights_file)

    # Now try to reuse these weights for a DIFFERENT grid
    ds_in_wrong = create_global_grid(20, 20)
    with pytest.raises(ValueError, match="Source grid shape"):
        Regridder(ds_in_wrong, ds_out, reuse_weights=True, filename=weights_file)


if __name__ == "__main__":
    pytest.main([__file__])
