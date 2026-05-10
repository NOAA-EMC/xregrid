import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_protocol_code_smells():
    """
    Verify that calling Regridder on a lazy DataArray does not trigger
    immediate computation of the data.
    """
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    # Check laziness: use a dask array with a delayed function that increments a counter
    from dask.delayed import delayed

    counter = [0]

    @delayed
    def count_calls(x):
        counter[0] += 1
        return x

    data = da.from_delayed(
        count_calls(np.random.rand(18, 36)), shape=(18, 36), dtype=float
    )
    da_lazy = xr.DataArray(
        data, dims=("lat", "lon"), coords={"lat": src.lat, "lon": src.lon}
    )

    regridder = Regridder(src, tgt)

    # This should NOT trigger count_calls
    res = regridder(da_lazy)

    assert counter[0] == 0, "Regridding triggered immediate computation!"

    # Computing the result SHOULD trigger it
    _ = res.compute()
    assert counter[0] > 0, "Computation did not trigger the delayed function!"


def test_extreme_coordinate_values():
    """Verify handling of coordinates slightly outside standard ranges."""
    # ESMF often fails if lat is exactly 90.000000000001
    # Our _clip_latitudes should handle this.
    src_lat = np.array([-90.000001, 0, 90.000001])
    src_lon = np.array([-0.000001, 180, 360.000001])

    src = xr.Dataset(coords={"lat": (["lat"], src_lat), "lon": (["lon"], src_lon)})
    src.lat.attrs["units"] = "degrees_north"
    src.lon.attrs["units"] = "degrees_east"

    tgt = create_global_grid(30, 30)

    # Should not raise ESMC_RC_ARG_OUTOFRANGE
    regridder = Regridder(src, tgt, method="bilinear")
    assert regridder is not None


def test_multiple_regridders_cache_isolation():
    """Verify that multiple regridder instances don't interfere via cache."""
    from conftest import setup_esmpy_mock
    from distributed import Client, LocalCluster

    with LocalCluster(n_workers=2, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            client.run(setup_esmpy_mock)

            src = create_global_grid(10, 10)
            tgt1 = create_global_grid(5, 5)
            tgt2 = create_global_grid(2, 2)

            r1 = Regridder(src, tgt1, parallel=True)
            r2 = Regridder(src, tgt2, parallel=True)

            assert r1._uid != r2._uid

            da = xr.DataArray(
                np.random.rand(18, 36),
                dims=("lat", "lon"),
                coords={"lat": src.lat, "lon": src.lon},
            )

            res1 = r1(da)
            res2 = r2(da)

            assert res1.shape == (36, 72)
            assert res2.shape == (90, 180)


if __name__ == "__main__":
    pytest.main([__file__])
