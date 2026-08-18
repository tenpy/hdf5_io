#pragma once

#include <highfive/highfive.hpp>
#include <pybind11/pybind11.h>
#include <string>

namespace hdf5_io {

namespace py = pybind11;

/// Integer HDF5 identifier stored by h5py (``obj.id.id``).
hid_t hid_from_h5py(py::handle obj);

/// True if `obj` is an h5py File, Group, or Dataset.
bool is_h5py_object(py::handle obj);
bool is_h5py_file(py::handle obj);
bool is_h5py_group(py::handle obj);
bool is_h5py_dataset(py::handle obj);

/// Borrow an h5py File/Group as a HighFive Group (increments the HDF5 refcount).
HighFive::Group wrap_group(py::handle obj);

/// Wrap an existing HDF5 object id back into an h5py Group or Dataset.
py::object h5py_from_hid(hid_t hid);

/// Open ``parent[path]`` as an h5py object after HighFive created it.
py::object h5py_getitem(py::handle parent, std::string const& path);

/// Create a subgroup via HighFive and return the h5py Group.
py::object h5_create_group(py::handle parent, std::string const& path);

/// Create a hard link ``parent[path] = src`` (src is an existing h5py Group/Dataset).
void h5_hard_link(py::handle parent, std::string const& path, py::handle src);

/// Write ``parent[path] = obj`` as a dataset (numpy/scalars/str/bytes).
void h5_write_dataset(py::handle parent, std::string const& path, py::handle obj);

/// Create a hard link or dataset, matching h5py ``parent[path] = obj``.
void h5_set_item(py::handle parent, std::string const& path, py::handle obj);

/// Set or overwrite an HDF5 attribute (string/int/float/bool/numpy scalar or array).
void h5_set_attr(py::handle h5obj, std::string const& name, py::handle value);

/// Read an attribute; decode bytes to str. Returns none if missing.
py::object h5_get_attr(py::handle h5obj, std::string const& name);

/// Whether ``name`` exists as a link in the group.
bool h5_contains(py::handle parent, std::string const& name);

} // namespace hdf5_io
