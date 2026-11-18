Input and output to and from hdf5
=================================

This repository is about the import and export features to HDF5 files included in `TeNPy <https://github.com/tenpy/tenpy>`_,
in particular the module ``tenpy.tools.hdf5_io``.
The introduction/specification of the format is at https://tenpy.readthedocs.io/en/latest/intro/input_output.html.

Since the general technique is not bound to the particular classes of TeNPy, I've separated the code into this repository.

The file ``src/python3/hdf5_io.py`` should just be the corresponding file of the ``tenpy.tools.hdf5_io`` module in `TeNPy`_.
It remains the reference Python implementation.

C++ implementation
------------------

The installable PyPI package ``tenpy-hdf5`` (import ``tenpy_hdf5``) provides the same public API via
pybind11 bindings (``tenpy_hdf5._core``). HDF5 groups, datasets, attributes, and hard links are
written through **libhdf5 + HighFive**, while callers still pass **h5py** ``File``/``Group``
objects. On-disk format and Python API are backwards compatible with the reference module.

The C++ library namespace and headers remain ``hdf5_io`` (``#include <hdf5_io/...>``).

The build is split into:

* ``hdf5_io_core`` — Python-free static library (``h5_ops``, ``core::Saver`` / ``core::Loader``,
  type registry). Link this from C++ projects such as lasyten (``HDF5_IO_BUILD_PYTHON=OFF``).
* ``hdf5_io_py`` + ``_core`` — pybind11 / h5py facade (optional, default ON for pip installs).

Build (editable install)::

    pip install -v --no-build-isolation -C editable.rebuild=true -e .

Core-only CMake (no Python)::

    cmake -S . -B build-core -DHDF5_IO_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
    cmake --build build-core && ctest --test-dir build-core

Requirements: Python >= 3.10, numpy, h5py, HDF5 development libraries, CMake, pybind11.
HighFive is fetched automatically via CMake if not installed system-wide.

Tests::

    pytest tests/

To make the export and import feature really usefull, we need scripts to convert the data such that it can be
imported/exported with other libraries except TeNPy.
We keep scripts for such a conversion in this repository, see the files ``src/python3/conversion*.py``.
