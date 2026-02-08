import xarray as xr
import numpy as np
from xregrid import Regridder

def test_descending_rectilinear():
    print("Testing descending rectilinear...")
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
    target_lat = np.arange(15, 76, 1.0)
    target_lon = np.arange(200, 331, 1.0)
    target_grid_ds = xr.Dataset(
        {
            "lat": (["lat"], target_lat, {"units": "degrees_north"}),
            "lon": (["lon"], target_lon, {"units": "degrees_east"}),
        }
    )

    # After my fix, this should have 0 zeros even without extrapolation
    regridder = Regridder(ds, target_grid_ds, method="bilinear")
    res = regridder(ds.air)

    zeros = np.sum(res.values == 0)
    print(f"Zeros: {zeros}")
    assert zeros == 0
    assert res.min() > 200
    print("Success!")

def test_non_monotonic():
    print("\nTesting non-monotonic...")
    # Source with shuffled coordinates
    lat = [40, 20, 30]
    lon = [200, 220, 210]
    data = np.array([[40, 42, 41], [20, 22, 21], [30, 32, 31]])
    src_ds = xr.Dataset(
        {"air": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon}
    )

    # Target (ordered)
    t_lat = [20, 30, 40]
    t_lon = [200, 210, 220]
    tgt_ds = xr.Dataset(coords={"lat": t_lat, "lon": t_lon})

    regridder = Regridder(src_ds, tgt_ds, method="bilinear")
    res = regridder(src_ds.air)

    print("Regridded values:")
    print(res.values)

    # Expected: identity after sorting
    expected = np.array([[20, 21, 22], [30, 31, 32], [40, 41, 42]])
    np.testing.assert_allclose(res.values, expected)
    print("Success!")

if __name__ == "__main__":
    try:
        test_descending_rectilinear()
    except Exception as e:
        print(f"Failed test_descending_rectilinear: {e}")

    try:
        test_non_monotonic()
    except Exception as e:
        print(f"Failed test_non_monotonic: {e}")
