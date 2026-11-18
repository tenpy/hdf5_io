#pragma once

#include <hdf5_io/h5_ops.h>
#include <hdf5_io/python.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace hdf5_io {

/// Integer HDF5 identifier stored by h5py (``obj.id.id``).
hid_t hid_from_h5py(py::handle obj);

/// True if `obj` is an h5py File, Group, or Dataset.
bool is_h5py_object(py::handle obj);
bool is_h5py_file(py::handle obj);
bool is_h5py_group(py::handle obj);
bool is_h5py_dataset(py::handle obj);

/// Borrow an h5py File/Group as a HighFive Group (increments the HDF5 refcount).
HighFive::Group wrap_group(py::handle obj);

/// Native object identity token for memoization (h5py overload).
std::string object_token(py::handle obj);

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
void h5_write_dataset(HighFive::Group& parent, std::string const& path, NumpyArray const& obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, py::bytes const& obj);

/// Create a hard link or dataset, matching h5py ``parent[path] = obj``.
void h5_set_item(py::handle parent, std::string const& path, py::handle obj);

/// Set or overwrite an HDF5 attribute (Python value).
void h5_set_attr(py::handle h5obj, std::string const& name, py::handle value);
void h5_set_attr(hid_t loc, std::string const& name, py::handle value);

/// Read an attribute; decode bytes to str. Returns none if missing.
py::object h5_get_attr(py::handle h5obj, std::string const& name);
py::object h5_get_attr(hid_t loc, std::string const& name);

/// Whether ``name`` exists as a link in the group.
bool h5_contains(py::handle parent, std::string const& name);

/// Read a dataset as a numpy array (0-d for scalar dataspace).
NumpyArray h5_read_array(hid_t dset);

} // namespace hdf5_io
