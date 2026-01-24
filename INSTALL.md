# Installation Instructions

There are two primary ways to set up the environment for `xregrid`.

## 1. Using Mamba (Recommended)

The easiest way to install all dependencies, including `esmpy` and `xesmf`, is using `mamba` or `micromamba` with the `conda-forge` channel.

```bash
# Create the environment from the provided yaml file
mamba env create -f environment.yml

# Activate the environment
mamba activate xregrid-env
```

## 2. Installing ESMPy from Source

If you need to build `esmpy` from source (e.g., for a specific ESMF version or custom build), follow these steps:

### Prerequisites
1.  **ESMF Library**: You must have the ESMF C++ library built and installed on your system.
2.  **Environment Variables**: Set `ESMF_DIR` to the path where ESMF is installed.
    ```bash
    export ESMF_DIR=/path/to/esmf
    ```
    Depending on your build, you might also need:
    ```bash
    export ESMF_COMPILER=gfortran
    export ESMF_COMM=openmpi
    ```

### Build and Install ESMPy
`esmpy` is located in the ESMF source tree under `src/addon/esmpy`.

```bash
# Navigate to the esmpy directory in the ESMF source
cd $ESMF_DIR/src/addon/esmpy

# Install from source
pip install .
```

### Install xregrid
Once `esmpy` is installed, you can install this package:

```bash
# From the root of this repository
pip install .
```

## Troubleshooting
- **Library Path**: Ensure that the ESMF shared libraries are in your `LD_LIBRARY_PATH`.
- **MPI**: If your ESMF was built with MPI, ensure that the corresponding MPI library is available at runtime.
