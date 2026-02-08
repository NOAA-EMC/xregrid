from __future__ import annotations

from typing import Any

from .utils import (
    create_global_grid,
    create_grid_from_crs,
    create_mesh_from_coords,
    create_regional_grid,
    load_esmf_file,
)
from .xregrid import Regridder

# Lazy loading of visualization functions to keep core regridding lean (Aero Protocol)
_VIZ_FUNCTIONS = [
    "plot",
    "plot_static",
    "plot_interactive",
    "plot_comparison",
    "plot_comparison_interactive",
    "plot_diagnostics",
    "plot_diagnostics_interactive",
]


def __getattr__(name: str) -> Any:
    if name in _VIZ_FUNCTIONS:
        import importlib

        viz = importlib.import_module(".viz", __name__)
        return getattr(viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Regridder",
    "create_global_grid",
    "create_regional_grid",
    "create_grid_from_crs",
    "create_mesh_from_coords",
    "load_esmf_file",
] + _VIZ_FUNCTIONS
