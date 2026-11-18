#pragma once

#include <tenpy_hdf5/python.h>
#include <tenpy_hdf5/tenpy_hdf5.h>

namespace tenpy_hdf5 {

void bind_tenpy_hdf5(py::module_& m);

} // namespace tenpy_hdf5
