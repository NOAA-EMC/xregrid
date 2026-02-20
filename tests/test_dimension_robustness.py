import numpy as np
import xarray as xr
from xregrid import Regridder


def test_regridder_time_dimension_detection():
    # Setup source and target grids with time
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    times = [np.datetime64("2020-01-01")]

    src_ds = xr.Dataset(
        coords={
            "time": (["time"], times, {"standard_name": "time"}),
            "lat": (
                ["time", "lat"],
                np.broadcast_to(lats, (1, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    # This should now work without failing during weight generation
    regridder = Regridder(src_ds, tgt_ds, method="bilinear")

    # Create data with time and vertical dimensions
    levs = np.arange(5)
    data = np.random.rand(len(times), len(levs), len(lats), len(lons))
    da = xr.DataArray(
        data,
        coords={
            "time": (["time"], times),
            "lev": (["lev"], levs),
            "lat": (["time", "lat"], np.broadcast_to(lats, (1, 10))),
            "lon": (["lon"], lons),
        },
        dims=("time", "lev", "lat", "lon"),
        name="temp",
    )

    # Regrid DataArray
    res_da = regridder(da)

    # Check that time and lev are preserved
    assert "time" in res_da.dims
    assert "lev" in res_da.dims
    assert res_da.shape == (1, 5, 5, 10)

    # Regrid Dataset
    ds = xr.Dataset({"temp": da, "time_var": (["time"], times)})
    res_ds = regridder(ds)

    assert "time" in res_ds.dims
    assert "temp" in res_ds.data_vars
    assert "time_var" in res_ds.data_vars
    assert res_ds["temp"].shape == (1, 5, 5, 10)
    assert res_ds["time_var"].dims == ("time",)


def test_regridder_dtype_time_fallback():
    # Setup with time-like dtype but non-standard name
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    times = [np.datetime64("2020-01-01")]

    src_ds = xr.Dataset(
        coords={
            "mytime": (["mytime"], times),  # Non-standard name, no CF attributes
            "lat": (
                ["mytime", "lat"],
                np.broadcast_to(lats, (1, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)

    # Verify mytime was detected as non-spatial
    assert "mytime" not in regridder._dims_source

    # Test DataArray regridding with this non-standard time dim
    da = xr.DataArray(
        np.random.rand(1, 10, 20), coords=src_ds.coords, dims=("mytime", "lat", "lon")
    )

    res = regridder(da)
    assert "mytime" in res.dims
    assert res.shape == (1, 5, 10)


def test_non_regriddable_object():
    # Test passing something that shouldn't be regridded
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)

    src_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lats,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )
    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)

    # A DataArray that only has one dimension (time)
    time_da = xr.DataArray([1, 2, 3], dims="time", name="time_var")

    # Should return unchanged
    res = regridder(time_da)
    xr.testing.assert_identical(res, time_da)


def test_regridder_vertical_dimension_detection():
    # Setup source with vertical dimension in lats
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    levs = np.arange(3)

    src_ds = xr.Dataset(
        coords={
            "lev": (["lev"], levs, {"standard_name": "altitude"}),
            "lat": (
                ["lev", "lat"],
                np.broadcast_to(lats, (3, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)
    assert "lev" not in regridder._dims_source

    da = xr.DataArray(
        np.random.rand(3, 10, 20), coords=src_ds.coords, dims=("lev", "lat", "lon")
    )

    res = regridder(da)
    assert "lev" in res.dims
    assert res.shape == (3, 5, 10)
