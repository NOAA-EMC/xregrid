import numpy as np
import xarray as xr
from xregrid import Regridder
from xregrid.utils import create_global_grid


def test_diagnostics_backend_provenance_eager():
    """
    Verify diagnostics() records backend=Eager for default (NumPy) regridding.
    """
    src = create_global_grid(10.0, 10.0, add_bounds=False)
    tgt = create_global_grid(5.0, 5.0, add_bounds=False)

    regridder = Regridder(src, tgt, method="bilinear", parallel=False)
    diag = regridder.diagnostics()

    assert isinstance(diag.weight_sum.data, np.ndarray)
    assert "backend=Eager" in diag.attrs["history"]


def test_quality_report_backend_provenance_eager():
    """
    Verify quality_report() records backend=Eager for default (NumPy) regridding.
    """
    src = create_global_grid(10.0, 10.0, add_bounds=False)
    tgt = create_global_grid(5.0, 5.0, add_bounds=False)

    regridder = Regridder(src, tgt, method="bilinear", parallel=False)
    report = regridder.quality_report(format="dataset")

    assert isinstance(report.unmapped_count.data, np.ndarray)
    assert "backend=Eager" in report.attrs["history"]


def test_regrid_provenance_backend_eager():
    """
    Verify regridding records backend=Eager for DataArray and Dataset.
    """
    src = create_global_grid(10.0, 10.0, add_bounds=False)
    tgt = create_global_grid(5.0, 5.0, add_bounds=False)

    data = xr.DataArray(
        np.ones((18, 36)),
        coords={"lat": src.lat, "lon": src.lon},
        dims=["lat", "lon"],
        name="test",
    )

    regridder = Regridder(src, tgt, method="bilinear")

    # DataArray
    out = regridder(data)
    assert "backend=Eager" in out.attrs["history"]

    # Dataset
    ds = xr.Dataset({"v1": data})
    out_ds = regridder(ds)
    assert "backend=Eager" in out_ds.attrs["history"]
