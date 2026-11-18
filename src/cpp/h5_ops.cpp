#include <tenpy_hdf5/h5_ops.h>

#include <hdf5.h>
#include <highfive/highfive.hpp>

#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenpy_hdf5 {

void
check_hdf5(herr_t status, char const* what)
{
    if (status < 0)
        throw Hdf5ExportError(std::string("HDF5 error: ") + what);
}

std::string
normalized_path(std::string const& path)
{
    if (path.size() > 1 && path.front() == '/')
        return path.substr(1);
    return path;
}

namespace {

template<typename T>
void
write_scalar(HighFive::Group& group, std::string const& path, T const& value)
{
    auto space = HighFive::DataSpace(HighFive::DataSpace::DataspaceType::dataspace_scalar);
    auto dset = group.createDataSet(path, space, HighFive::create_datatype<T>());
    dset.write_raw(&value);
}

void
write_vlen_string(hid_t loc, std::string const& path, std::string const& value, bool utf8)
{
    hid_t type = H5Tcopy(H5T_C_S1);
    check_hdf5(H5Tset_size(type, H5T_VARIABLE), "H5Tset_size");
    check_hdf5(H5Tset_cset(type, utf8 ? H5T_CSET_UTF8 : H5T_CSET_ASCII), "H5Tset_cset");
    check_hdf5(H5Tset_strpad(type, H5T_STR_NULLTERM), "H5Tset_strpad");
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dset = H5Dcreate2(loc, path.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        H5Sclose(space);
        H5Tclose(type);
        throw Hdf5ExportError("failed to create string dataset " + path);
    }
    char const* ptr = value.c_str();
    herr_t st = H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ptr);
    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(type);
    check_hdf5(st, "H5Dwrite string");
}

void
write_attr_vlen_string(hid_t loc, std::string const& name, std::string const& value, bool utf8)
{
    if (H5Aexists(loc, name.c_str()) > 0)
        check_hdf5(H5Adelete(loc, name.c_str()), "H5Adelete");
    hid_t type = H5Tcopy(H5T_C_S1);
    check_hdf5(H5Tset_size(type, H5T_VARIABLE), "H5Tset_size");
    check_hdf5(H5Tset_cset(type, utf8 ? H5T_CSET_UTF8 : H5T_CSET_ASCII), "H5Tset_cset");
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        H5Tclose(type);
        throw Hdf5ExportError("failed to create attribute " + name);
    }
    char const* ptr = value.c_str();
    herr_t st = H5Awrite(attr, type, &ptr);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
    check_hdf5(st, "H5Awrite string");
}

template<typename T>
void
write_typed_buffer(HighFive::Group& group, std::string const& path, Hdf5Buffer const& buf)
{
    HighFive::DataSpace space =
      buf.shape.empty() ? HighFive::DataSpace(HighFive::DataSpace::DataspaceType::dataspace_scalar)
                        : HighFive::DataSpace(buf.shape);
    auto dset = group.createDataSet(path, space, HighFive::create_datatype<T>());
    std::size_t n = 1;
    for (auto d : buf.shape)
        n *= d;
    if (n > 0 || buf.shape.empty())
        dset.write_raw(static_cast<T const*>(buf.data()));
}

void
write_buffer(HighFive::Group& group, std::string const& path, Hdf5Buffer const& buf)
{
    if (buf.kind == 'f' && buf.itemsize == 4)
        return write_typed_buffer<float>(group, path, buf);
    if (buf.kind == 'f' && buf.itemsize == 8)
        return write_typed_buffer<double>(group, path, buf);
    if (buf.kind == 'c' && buf.itemsize == 8)
        return write_typed_buffer<std::complex<float>>(group, path, buf);
    if (buf.kind == 'c' && buf.itemsize == 16)
        return write_typed_buffer<std::complex<double>>(group, path, buf);
    if (buf.kind == 'i' && buf.itemsize == 1)
        return write_typed_buffer<std::int8_t>(group, path, buf);
    if (buf.kind == 'i' && buf.itemsize == 2)
        return write_typed_buffer<std::int16_t>(group, path, buf);
    if (buf.kind == 'i' && buf.itemsize == 4)
        return write_typed_buffer<std::int32_t>(group, path, buf);
    if (buf.kind == 'i' && buf.itemsize == 8)
        return write_typed_buffer<std::int64_t>(group, path, buf);
    if (buf.kind == 'u' && buf.itemsize == 1)
        return write_typed_buffer<std::uint8_t>(group, path, buf);
    if (buf.kind == 'u' && buf.itemsize == 2)
        return write_typed_buffer<std::uint16_t>(group, path, buf);
    if (buf.kind == 'u' && buf.itemsize == 4)
        return write_typed_buffer<std::uint32_t>(group, path, buf);
    if (buf.kind == 'u' && buf.itemsize == 8)
        return write_typed_buffer<std::uint64_t>(group, path, buf);
    if (buf.kind == 'b')
        return write_typed_buffer<std::uint8_t>(group, path, buf);
    throw Hdf5ExportError("unsupported buffer dtype for HDF5 dataset");
}

bool
is_hdf5_bool_enum(hid_t ntype)
{
    if (H5Tget_class(ntype) != H5T_ENUM)
        return false;
    int n = H5Tget_nmembers(ntype);
    if (n != 2)
        return false;
    char* a = H5Tget_member_name(ntype, 0);
    char* b = H5Tget_member_name(ntype, 1);
    bool ok = false;
    if (a && b) {
        std::string na(a);
        std::string nb(b);
        ok = (na == "FALSE" && nb == "TRUE") || (na == "TRUE" && nb == "FALSE");
    }
    if (a)
        H5free_memory(a);
    if (b)
        H5free_memory(b);
    return ok;
}

void
dtype_from_hdf5(hid_t ntype, char& kind, int& itemsize)
{
    H5T_class_t cls = H5Tget_class(ntype);
    size_t sz = H5Tget_size(ntype);
    itemsize = static_cast<int>(sz);
    if (cls == H5T_ENUM) {
        if (is_hdf5_bool_enum(ntype)) {
            kind = 'b';
            itemsize = 1;
            return;
        }
        hid_t super = H5Tget_super(ntype);
        dtype_from_hdf5(super, kind, itemsize);
        H5Tclose(super);
        return;
    }
    if (cls == H5T_BITFIELD) {
        kind = 'u';
        return;
    }
    if (cls == H5T_INTEGER) {
        kind = (H5Tget_sign(ntype) == H5T_SGN_NONE) ? 'u' : 'i';
        return;
    }
    if (cls == H5T_FLOAT) {
        kind = 'f';
        return;
    }
    if (cls == H5T_COMPOUND) {
        kind = 'c';
        return;
    }
    throw Hdf5ExportError("unsupported HDF5 dataset dtype");
}

} // namespace

HighFive::Group
group_from_hid(hid_t hid)
{
    return HighFive::detail::make_group(hid);
}

std::string
object_token(hid_t hid)
{
#if H5_VERSION_GE(1, 12, 0)
    H5O_info2_t info{};
    check_hdf5(H5Oget_info3(hid, &info, H5O_INFO_BASIC), "H5Oget_info3");
    char* token_cstr = nullptr;
    check_hdf5(H5Otoken_to_str(hid, &info.token, &token_cstr), "H5Otoken_to_str");
    std::string token(token_cstr);
    check_hdf5(H5free_memory(token_cstr), "H5free_memory");
    return token;
#else
    // HDF5 < 1.12 (e.g. manylinux EPEL 1.10.5): identity is (fileno, addr).
    H5O_info_t info{};
    check_hdf5(H5Oget_info(hid, &info), "H5Oget_info");
    return std::to_string(info.fileno) + ":" +
           std::to_string(static_cast<unsigned long long>(info.addr));
#endif
}

HighFive::Group
h5_create_group(HighFive::Group& parent, std::string const& path)
{
    return parent.createGroup(normalized_path(path));
}

void
h5_hard_link(HighFive::Group& parent, std::string const& path, hid_t src)
{
    herr_t st = H5Lcreate_hard(
      src, ".", parent.getId(), normalized_path(path).c_str(), H5P_DEFAULT, H5P_DEFAULT);
    check_hdf5(st, "H5Lcreate_hard");
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, std::string const& obj)
{
    write_vlen_string(parent.getId(), normalized_path(path), obj, true);
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, std::int64_t obj)
{
    write_scalar<std::int64_t>(parent, normalized_path(path), obj);
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, std::uint64_t obj)
{
    write_scalar<std::uint64_t>(parent, normalized_path(path), obj);
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, double obj)
{
    write_scalar<double>(parent, normalized_path(path), obj);
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, std::complex<double> const& obj)
{
    write_scalar<std::complex<double>>(parent, normalized_path(path), obj);
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, bool obj)
{
    write_scalar<std::uint8_t>(parent, normalized_path(path), obj ? 1u : 0u);
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, Hdf5Buffer const& buf)
{
    write_buffer(parent, normalized_path(path), buf);
}

void
h5_set_item(HighFive::Group& parent, std::string const& path, hid_t src)
{
    h5_hard_link(parent, path, src);
}

hid_t
h5_open(hid_t loc, std::string const& path)
{
    if (path.empty() || path == "/") {
        check_hdf5(H5Iinc_ref(loc), "H5Iinc_ref");
        return loc;
    }
    hid_t id = H5Oopen(loc, path.c_str(), H5P_DEFAULT);
    if (id < 0)
        throw Hdf5ExportError("failed to open HDF5 object " + path);
    return id;
}

hid_t
h5_open(HighFive::Group& parent, std::string const& path)
{
    return h5_open(parent.getId(), path);
}

std::string
h5_object_name(hid_t hid)
{
    ssize_t sz = H5Iget_name(hid, nullptr, 0);
    if (sz < 0)
        throw Hdf5ExportError("H5Iget_name failed");
    if (sz == 0)
        return "/";
    std::string name(static_cast<size_t>(sz), '\0');
    H5Iget_name(hid, name.data(), static_cast<size_t>(sz) + 1);
    return name;
}

bool
h5_contains(hid_t loc, std::string const& name)
{
    htri_t exists = H5Lexists(loc, name.c_str(), H5P_DEFAULT);
    return exists > 0;
}

bool
h5_contains(HighFive::Group& parent, std::string const& name)
{
    return h5_contains(parent.getId(), name);
}

std::vector<std::string>
h5_link_names(hid_t loc)
{
    H5G_info_t info{};
    check_hdf5(H5Gget_info(loc, &info), "H5Gget_info");
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(info.nlinks));
    for (hsize_t i = 0; i < info.nlinks; ++i) {
        ssize_t sz =
          H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        if (sz < 0)
            throw Hdf5ExportError("H5Lget_name_by_idx failed");
        std::string name(static_cast<size_t>(sz), '\0');
        H5Lget_name_by_idx(loc,
                           ".",
                           H5_INDEX_NAME,
                           H5_ITER_INC,
                           i,
                           name.data(),
                           static_cast<size_t>(sz) + 1,
                           H5P_DEFAULT);
        names.push_back(std::move(name));
    }
    return names;
}

void
h5_set_attr(hid_t loc, std::string const& name, std::string const& value)
{
    write_attr_vlen_string(loc, name, value, true);
}

void
h5_set_attr(hid_t loc, std::string const& name, std::int64_t value)
{
    if (H5Aexists(loc, name.c_str()) > 0)
        check_hdf5(H5Adelete(loc, name.c_str()), "H5Adelete");
    hid_t type = H5Tcopy(H5T_NATIVE_INT64);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        H5Tclose(type);
        throw Hdf5ExportError("failed to create attribute " + name);
    }
    herr_t st = H5Awrite(attr, type, &value);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
    check_hdf5(st, "H5Awrite int64");
}

void
h5_set_attr(hid_t loc, std::string const& name, bool value)
{
    if (H5Aexists(loc, name.c_str()) > 0)
        check_hdf5(H5Adelete(loc, name.c_str()), "H5Adelete");
    hid_t type = H5Tcopy(H5T_NATIVE_UINT8);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        H5Tclose(type);
        throw Hdf5ExportError("failed to create attribute " + name);
    }
    std::uint8_t v = value ? 1u : 0u;
    herr_t st = H5Awrite(attr, type, &v);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
    check_hdf5(st, "H5Awrite bool");
}

std::optional<std::string>
h5_get_attr_string(hid_t loc, std::string const& name)
{
    if (H5Aexists(loc, name.c_str()) <= 0)
        return std::nullopt;
    hid_t attr = H5Aopen(loc, name.c_str(), H5P_DEFAULT);
    if (attr < 0)
        throw Hdf5ExportError("failed to open attribute " + name);
    hid_t type = H5Aget_type(attr);
    H5T_class_t cls = H5Tget_class(type);
    std::optional<std::string> result;
    if (cls == H5T_STRING) {
        if (H5Tis_variable_str(type) > 0) {
            char* buf = nullptr;
            check_hdf5(H5Aread(attr, type, &buf), "H5Aread string");
            result = std::string(buf ? buf : "");
            if (buf)
                H5free_memory(buf);
        } else {
            size_t sz = H5Tget_size(type);
            std::string s(sz, '\0');
            check_hdf5(H5Aread(attr, type, s.data()), "H5Aread string");
            auto z = s.find('\0');
            if (z != std::string::npos)
                s.resize(z);
            result = std::move(s);
        }
    }
    H5Tclose(type);
    H5Aclose(attr);
    return result;
}

std::optional<std::int64_t>
h5_get_attr_int64(hid_t loc, std::string const& name)
{
    if (H5Aexists(loc, name.c_str()) <= 0)
        return std::nullopt;
    hid_t attr = H5Aopen(loc, name.c_str(), H5P_DEFAULT);
    if (attr < 0)
        throw Hdf5ExportError("failed to open attribute " + name);
    hid_t type = H5Aget_type(attr);
    long long v = 0;
    hid_t ntype = H5Tcopy(H5T_NATIVE_LLONG);
    check_hdf5(H5Aread(attr, ntype, &v), "H5Aread int");
    H5Tclose(ntype);
    H5Tclose(type);
    H5Aclose(attr);
    return static_cast<std::int64_t>(v);
}

std::string
h5_read_vlen_string(hid_t dset)
{
    hid_t type = H5Dget_type(dset);
    std::string result;
    if (H5Tis_variable_str(type) > 0) {
        char* buf = nullptr;
        check_hdf5(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buf), "H5Dread vlen");
        result = buf ? buf : "";
        if (buf)
            H5free_memory(buf);
    } else {
        size_t sz = H5Tget_size(type);
        result.assign(sz, '\0');
        check_hdf5(H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()),
                   "H5Dread string");
        auto z = result.find('\0');
        if (z != std::string::npos)
            result.resize(z);
    }
    H5Tclose(type);
    return result;
}

Hdf5Buffer
h5_read_buffer(hid_t dset)
{
    hid_t ftype = H5Dget_type(dset);
    hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
    hid_t space = H5Dget_space(dset);
    Hdf5Buffer buf;
    dtype_from_hdf5(ntype, buf.kind, buf.itemsize);
    H5S_class_t space_type = H5Sget_simple_extent_type(space);
    if (space_type == H5S_SCALAR) {
        buf.shape.clear();
    } else {
        int ndims = H5Sget_simple_extent_ndims(space);
        std::vector<hsize_t> dims(static_cast<size_t>(ndims));
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        buf.shape.assign(dims.begin(), dims.end());
    }
    std::size_t nbytes = buf.nbytes();
    buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[nbytes ? nbytes : 1]);
    if (nbytes > 0 || space_type == H5S_SCALAR)
        check_hdf5(H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.storage.get()),
                   "H5Dread array");
    H5Sclose(space);
    H5Tclose(ntype);
    H5Tclose(ftype);
    return buf;
}

} // namespace tenpy_hdf5
