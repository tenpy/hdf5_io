#include "py_tenpy_hdf5.h"

NB_MODULE(_core, m)
{
    m.doc() = "C++ tenpy_hdf5 bindings (TeNPy HDF5 format)";
    tenpy_hdf5::bind_tenpy_hdf5(m);
}
