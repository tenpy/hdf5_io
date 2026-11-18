#pragma once

#include <tenpy_hdf5/tenpy_hdf5.h>
#include <tenpy_hdf5/python.h>

namespace tenpy_hdf5 {

void bind_tenpy_hdf5(py::module_& m);

} // namespace tenpy_hdf5
