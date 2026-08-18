#include "py_hdf5_io.h"

PYBIND11_MODULE(_core, m)
{
    m.doc() = "C++ hdf5_io bindings (TeNPy HDF5 format)";
    hdf5_io::bind_hdf5_io(m);
}
