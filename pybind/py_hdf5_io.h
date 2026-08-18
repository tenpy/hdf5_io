#pragma once

#include <hdf5_io/hdf5_io.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hdf5_io {

void bind_hdf5_io(py::module_& m);

} // namespace hdf5_io
