#pragma once

/// Python-free HDF5/HighFive helpers used by tenpy_hdf5_core and the nanobind bridge.

#include <tenpy_hdf5/exceptions.h>

#include <complex>
#include <cstdint>
#include <highfive/highfive.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tenpy_hdf5 {

/// Contiguous array view for dataset I/O.
struct Hdf5Buffer
{
    std::shared_ptr<std::uint8_t[]> storage;
    std::vector<std::size_t> shape;
    /// HDF5 / numpy-style kind: 'f','i','u','c','b' plus itemsize in bytes.
    char kind = 'f';
    int itemsize = 8;

    void const* data() const { return storage.get(); }
    void* mutable_data() { return storage.get(); }
    std::size_t nbytes() const
    {
        std::size_t n = static_cast<std::size_t>(itemsize);
        if (shape.empty())
            return n;
        for (auto d : shape)
            n *= d;
        return n;
    }
};

/// Normalize a relative path (strip leading '/').
std::string normalized_path(std::string const& path);

void check_hdf5(herr_t status, char const* what);

/// Own a group hid as HighFive::Group (takes ownership of `hid`).
HighFive::Group group_from_hid(hid_t hid);

/// Native object identity token for memoization across reopened handles and hard links.
std::string object_token(hid_t hid);

/// HDF5 object name (full path in the file).
std::string h5_object_name(hid_t hid);

/// Open a child group or dataset; caller owns the returned hid (`H5Idec_ref`).
hid_t h5_open(HighFive::Group& parent, std::string const& path);
hid_t h5_open(hid_t loc, std::string const& path);

HighFive::Group h5_create_group(HighFive::Group& parent, std::string const& path);

void h5_hard_link(HighFive::Group& parent, std::string const& path, hid_t src);

void h5_set_item(HighFive::Group& parent, std::string const& path, hid_t src);

void h5_write_dataset(HighFive::Group& parent, std::string const& path, std::string const& obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, std::int64_t obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, std::uint64_t obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, double obj);
void h5_write_dataset(HighFive::Group& parent,
                      std::string const& path,
                      std::complex<double> const& obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, bool obj);
void h5_write_dataset(HighFive::Group& parent, std::string const& path, Hdf5Buffer const& buf);

void h5_set_attr(hid_t loc, std::string const& name, std::string const& value);
void h5_set_attr(hid_t loc, std::string const& name, std::int64_t value);
void h5_set_attr(hid_t loc, std::string const& name, bool value);

/// Read a string attribute; empty optional if missing.
std::optional<std::string> h5_get_attr_string(hid_t loc, std::string const& name);
std::optional<std::int64_t> h5_get_attr_int64(hid_t loc, std::string const& name);

bool h5_contains(HighFive::Group& parent, std::string const& name);
bool h5_contains(hid_t loc, std::string const& name);

std::vector<std::string> h5_link_names(hid_t loc);

std::string h5_read_vlen_string(hid_t dset);

/// Read a numeric/bool dataset into an owned buffer (fills shape/kind/itemsize).
Hdf5Buffer h5_read_buffer(hid_t dset);

} // namespace tenpy_hdf5
