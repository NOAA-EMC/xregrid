from __future__ import annotations

from typing import Any

try:
    import esmpy
except ImportError:
    esmpy = None


def get_regrid_method_map() -> dict[str, Any]:
    """
    Get the mapping of string names to ESMF RegridMethod constants.

    Returns
    -------
    dict
        Mapping of method names to ESMF constants.
    """
    if esmpy is None:
        return {}
    return {
        "bilinear": esmpy.RegridMethod.BILINEAR,
        "conservative": esmpy.RegridMethod.CONSERVE,
        "nearest_s2d": esmpy.RegridMethod.NEAREST_STOD,
        "nearest_d2s": esmpy.RegridMethod.NEAREST_DTOS,
        "patch": esmpy.RegridMethod.PATCH,
    }


def get_extrap_method_map() -> dict[str, Any]:
    """
    Get the mapping of string names to ESMF ExtrapMethod constants.

    Returns
    -------
    dict
        Mapping of extrapolation names to ESMF constants.
    """
    if esmpy is None:
        return {}
    return {
        "nearest_s2d": esmpy.ExtrapMethod.NEAREST_STOD,
        "nearest_idw": esmpy.ExtrapMethod.NEAREST_IDAVG,
        "creep_fill": esmpy.ExtrapMethod.CREEP_FILL,
    }


def get_coord_sys(name: str = "SPH_DEG") -> Any | None:
    """
    Get an ESMF CoordSys constant by name.

    Parameters
    ----------
    name : str, default 'SPH_DEG'
        The name of the coordinate system ('SPH_DEG' or 'CART').

    Returns
    -------
    esmpy.CoordSys or None
        The ESMF CoordSys constant, or None if esmpy is not installed.
    """
    if esmpy is None:
        return None
    if name == "SPH_DEG":
        return esmpy.CoordSys.SPH_DEG
    elif name == "CART":
        return esmpy.CoordSys.CART
    return None
