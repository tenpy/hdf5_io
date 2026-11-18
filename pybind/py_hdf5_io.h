#pragma once

#include <hdf5_io/hdf5_io.h>
#include <hdf5_io/python.h>

namespace hdf5_io {

void bind_hdf5_io(py::module_& m);

} // namespace hdf5_io
