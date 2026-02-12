import os
import importlib.util
from setuptools import setup

# Use tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        # Fallback for basic installations during bootstrap if tomli is not yet available
        tomllib = None


def get_install_requires():
    """Read dependencies from pyproject.toml and adjust for ESMF environment."""
    # Static fallback list in case pyproject.toml cannot be parsed
    # Must be kept in sync with the primary list in pyproject.toml
    default_deps = [
        "xarray",
        "numpy",
        "scipy",
        "dask",
        "netCDF4",
        "esmpy",
        "cf-xarray",
        "pyproj",
    ]

    if not os.path.exists("pyproject.toml") or tomllib is None:
        deps = default_deps
    else:
        try:
            with open("pyproject.toml", "rb") as f:
                data = tomllib.load(f)
            # Read from the custom [tool.xregrid] section
            deps = (
                data.get("tool", {})
                .get("xregrid", {})
                .get("dependencies", default_deps)
            )
        except Exception:
            deps = default_deps

    # ESMFMKFILE Support:
    # If ESMFMKFILE is set, it means the user has a pre-existing ESMF installation.
    # We check if esmpy is already available. If not, and ESMFMKFILE is set,
    # we remove esmpy from install_requires to prevent pip from failing with
    # 'DistributionNotFound' (since esmpy is often not on PyPI for all platforms).
    # This allows the user to install esmpy manually from source using ESMFMKFILE.
    if os.environ.get("ESMFMKFILE"):
        # Use find_spec to avoid risky imports of shared libraries during setup
        esmpy_installed = importlib.util.find_spec("esmpy") is not None
        if not esmpy_installed:
            if "esmpy" in deps:
                print("\n" + "=" * 80)
                print("NOTICE: ESMFMKFILE detected but esmpy is not installed.")
                print("To support your pre-existing ESMF installation, we are omitting")
                print("the 'esmpy' requirement from the automatic installation.")
                print("Please install esmpy manually from the ESMF source tree:")
                print("  cd $ESMF_DIR/src/addon/esmpy && python setup.py install")
                print("=" * 80 + "\n")
                deps = [d for d in deps if d != "esmpy"]

    return deps


if __name__ == "__main__":
    setup(
        install_requires=get_install_requires(),
    )
