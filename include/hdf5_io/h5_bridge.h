#pragma once

#include <complex>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

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

/// Own a group hid as HighFive::Group (takes ownership of `hid`).
HighFive::Group group_from_hid(hid_t hid);

/// Native object identity token for memoization across reopened handles and hard links.
std::string object_token(hid_t hid);
std::string object_token(py::handle obj);

/// HDF5 object name (full path in the file).
std::string h5_object_name(hid_t hid);

/// Wrap an existing HDF5 object id back into an h5py Group or Dataset.
py::object h5py_from_hid(hid_t hid);

/// Open ``parent[path]`` as an h5py object after HighFive created it.
py::object h5py_getitem(py::handle parent, std::string const& path);

/// Open a child group or dataset; caller owns the returned hid (`H5Idec_ref`).
hid_t h5_open(HighFive::Group& parent, std::string const& path);
hid_t h5_open(hid_t loc, std::string const& path);

/// Create a subgroup via HighFive and return the h5py Group.
py::object h5_create_group(py::handle parent, std::string const& path);
HighFive::Group h5_create_group(HighFive::Group& parent, std::string const& path);

/// Create a hard link ``parent[path] = src`` (src is an existing h5py Group/Dataset).
void h5_hard_link(py::handle parent, std::string const& path, py::handle src);
void h5_hard_link(HighFive::Group& parent, std::string const& path, hid_t src);

/// Write ``parent[path] = obj`` as a dataset (numpy/scalars/str/bytes).
void h5_write_dataset(py::handle parent, std::string const& path, py::handle obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, py::array const& obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, std::string const& obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, py::bytes const& obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, std::int64_t obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, std::uint64_t obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, double obj);
void h5_write_dataset(HighFive::Group& parent,
                      std::string const& path,
                      std::complex<double> const& obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, bool obj);

/// Create a hard link or dataset, matching h5py ``parent[path] = obj``.
void h5_set_item(py::handle parent, std::string const& path, py::handle obj);
void h5_set_item(HighFive::Group& parent, std::string const& path, hid_t src);

/// Set or overwrite an HDF5 attribute.
void h5_set_attr(py::handle h5obj, std::string const& name, py::handle value);
void h5_set_attr(hid_t loc, std::string const& name, py::handle value);
void h5_set_attr(hid_t loc, std::string const& name, std::string const& value);
void h5_set_attr(hid_t loc, std::string const& name, std::int64_t value);
void h5_set_attr(hid_t loc, std::string const& name, bool value);

/// Read an attribute; decode bytes to str. Returns none if missing.
py::object h5_get_attr(py::handle h5obj, std::string const& name);
py::object h5_get_attr(hid_t loc, std::string const& name);

/// Whether ``name`` exists as a link in the group.
bool h5_contains(py::handle parent, std::string const& name);
bool h5_contains(HighFive::Group& parent, std::string const& name);
bool h5_contains(hid_t loc, std::string const& name);

/// Link names in a group (non-recursive).
std::vector<std::string> h5_link_names(hid_t loc);

/// Read a dataset as a numpy array (0-d for scalar dataspace).
py::array h5_read_array(hid_t dset);

/// Read a scalar/vlen string dataset.
std::string h5_read_vlen_string(hid_t dset);

} // namespace hdf5_io
