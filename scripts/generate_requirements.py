import os
import sys

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: tomllib or tomli is required to run this script.")
        sys.exit(1)


def main():
    if not os.path.exists("pyproject.toml"):
        print("Error: pyproject.toml not found.")
        sys.exit(1)

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    # Try tool section first (new structure), then project section (old structure)
    deps = data.get("tool", {}).get("xregrid", {}).get("dependencies")
    if deps is None:
        deps = data.get("project", {}).get("dependencies", [])

    # Core requirements
    with open("requirements.txt", "w") as f:
        f.write(
            "# This file is auto-generated from pyproject.toml. Do not edit directly.\n"
        )
        for dep in deps:
            f.write(f"{dep}\n")
    print("Successfully generated requirements.txt")

    # Custom requirements without esmpy (useful for custom ESMF builds)
    with open("requirements_no_esmpy.txt", "w") as f:
        f.write(
            "# This file is auto-generated from pyproject.toml. Do not edit directly.\n"
        )
        f.write("# It omits esmpy to support custom ESMF installations.\n")
        for dep in deps:
            if dep != "esmpy":
                f.write(f"{dep}\n")
    print("Successfully generated requirements_no_esmpy.txt")

    # Optional requirements (e.g. test, viz)
    optional_deps = data.get("project", {}).get("optional-dependencies", {})
    if not optional_deps:
        optional_deps = (
            data.get("tool", {}).get("xregrid", {}).get("optional-dependencies", {})
        )

    for extra, extra_deps in optional_deps.items():
        if extra == "full":
            continue  # Skip meta-extra

        filename = f"requirements-{extra}.txt"
        with open(filename, "w") as f:
            f.write(
                f"# This file is auto-generated from pyproject.toml [{extra}]. Do not edit directly.\n"
            )
            for dep in extra_deps:
                # If it's a reference to the package itself, like xregrid[test], skip or handle
                if dep.startswith("xregrid["):
                    continue
                f.write(f"{dep}\n")
        print(f"Successfully generated {filename}")


if __name__ == "__main__":
    main()
