Input and output to and from hdf5
=================================

This repository is about the import and export features to HDF5 files included in `TeNPy <https://github.com/tenpy/tenpy>`_,
in particular the module ``tenpy.tools.hdf5_io``.
The introduction/specification of the format is at https://tenpy.readthedocs.io/en/latest/intro/input_output.html.

Since the general technique is not bound to the particular classes of TeNPy, I've separated the code into this repository.

The file ``src/python3/hdf5_io.py`` should just be the corresponding file of the ``tenpy.tools.hdf5_io`` module in `TeNPy`_ (version 1.X).
It remains the reference Python implementation.

C++ implementation
------------------

The installable PyPI package ``tenpy-hdf5`` (``import tenpy_hdf5`` in python) provides the same public API via
nanobind bindings (``tenpy_hdf5._core``). HDF5 groups, datasets, attributes, and hard links are
written through **libhdf5 + HighFive**, while callers still pass **h5py** ``File``/``Group``
objects. On-disk format and Python API are backwards compatible with the reference module.

The C++ library namespace and headers are ``tenpy_hdf5`` (``#include <tenpy_hdf5/...>``).

The build is split into:

* ``tenpy_hdf5_core`` — Python-free static library (``h5_ops``, ``core::Saver`` / ``core::Loader``,
  type registry). Link this from C++ projects such as lasyten (``TENPY_HDF5_BUILD_PYTHON=OFF``).
* ``tenpy_hdf5_py`` + ``_core`` — nanobind / h5py facade (optional, default ON for pip installs).

Build (editable install)::

    pip install -v --no-build-isolation -C editable.rebuild=true -e .

Core-only CMake (no Python)::

    cmake -S . -B build-core -DTENPY_HDF5_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
    cmake --build build-core && ctest --test-dir build-core

Requirements: Python >= 3.12, numpy, h5py, HDF5 development libraries, CMake, nanobind >= 3.0.
HighFive is fetched automatically via CMake if not installed system-wide.
Wheels use the Python Stable ABI (``cp312``); see ``.github/workflows/publish.yml``.

**Same libhdf5 as h5py.** The C++ bindings take raw HDF5 identifiers from h5py
``File``/``Group`` objects, so ``tenpy_hdf5`` and ``h5py`` must be linked against the
*same* ``libhdf5``. Pip ``h5py`` wheels vendor their own HDF5 and will not work with an
extension built against system/conda HDF5. Prefer the conda-forge stack in
``environment.yml``, or build h5py against that same library
(``pip install --no-binary=h5py h5py``).

Tests
-----

Three entry points (all exercised in CI):

1. **Reference pure-Python** (TeNPy v1 sync; no C++ build)::

       cd src/python3 && PYTHONPATH=. pytest test_hdf5.py -vv

   Committed import fixtures: ``src/python3/data/*.hdf5``. Regenerate with::

       cd src/python3 && PYTHONPATH=. python test_hdf5.py

2. **Installable ``tenpy_hdf5`` package** (nanobind + cross-compat with the reference)::

       # Prefer conda-forge hdf5+h5py from environment.yml (same libhdf5).
       pip install --no-build-isolation -e ".[test]"
       pytest tests/ -vv
       ctest --test-dir build --output-on-failure

   Committed import fixtures: ``tests/data/*.hdf5`` (written with ``tenpy_hdf5``
   so ``ATTR_MODULE`` resolves to the installable package).

3. **Python-free core** (C++ consumers such as lasyten)::

       cmake -S . -B build-core -DTENPY_HDF5_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
       cmake --build build-core && ctest --test-dir build-core

``src/python2/`` is kept for conversion scripts only and is not in CI.

To make the export and import feature really usefull, we need scripts to convert the data such that it can be
imported/exported with other libraries except TeNPy.
We keep scripts for such a conversion in this repository, see the files ``src/python3/conversion*.py``.
