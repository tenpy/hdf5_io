#include <hdf5_io/exceptions.h>
#include <hdf5_io/h5_bridge.h>

#include <hdf5.h>
#include <highfive/highfive.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace hdf5_io {

namespace {

py::module_
h5py_mod()
{
    return py::module_::import("h5py");
}

void
check_hdf5(herr_t status, char const* what)
{
    if (status < 0)
        throw Hdf5ExportError(std::string("HDF5 error: ") + what);
}

HighFive::DataSpace
numpy_space(py::array const& arr)
{
    if (arr.ndim() == 0)
        return HighFive::DataSpace(HighFive::DataSpace::DataspaceType::dataspace_scalar);
    std::vector<size_t> dims(static_cast<size_t>(arr.ndim()));
    for (py::ssize_t i = 0; i < arr.ndim(); ++i)
        dims[static_cast<size_t>(i)] = static_cast<size_t>(arr.shape(i));
    return HighFive::DataSpace(dims);
}

py::array
as_c_contiguous(py::array arr)
{
    if (arr.attr("flags").attr("c_contiguous").cast<bool>())
        return arr;
    return py::reinterpret_steal<py::array>(arr.attr("copy")("C").release());
}

template<typename T>
void
write_typed_array(HighFive::Group& group, std::string const& path, py::array arr)
{
    arr = as_c_contiguous(arr);
    auto space = numpy_space(arr);
    auto dset = group.createDataSet(path, space, HighFive::create_datatype<T>());
    if (arr.size() > 0 || arr.ndim() == 0)
        dset.write_raw(static_cast<T const*>(arr.data()));
}

void
write_numpy_array(HighFive::Group& group, std::string const& path, py::array arr)
{
    py::dtype dt = arr.dtype();
    char kind = dt.kind();
    int itemsize = static_cast<int>(dt.itemsize());
    if (kind == 'f' && itemsize == 4)
        return write_typed_array<float>(group, path, arr);
    if (kind == 'f' && itemsize == 8)
        return write_typed_array<double>(group, path, arr);
    if (kind == 'c' && itemsize == 8)
        return write_typed_array<std::complex<float>>(group, path, arr);
    if (kind == 'c' && itemsize == 16)
        return write_typed_array<std::complex<double>>(group, path, arr);
    if (kind == 'i' && itemsize == 1)
        return write_typed_array<std::int8_t>(group, path, arr);
    if (kind == 'i' && itemsize == 2)
        return write_typed_array<std::int16_t>(group, path, arr);
    if (kind == 'i' && itemsize == 4)
        return write_typed_array<std::int32_t>(group, path, arr);
    if (kind == 'i' && itemsize == 8)
        return write_typed_array<std::int64_t>(group, path, arr);
    if (kind == 'u' && itemsize == 1)
        return write_typed_array<std::uint8_t>(group, path, arr);
    if (kind == 'u' && itemsize == 2)
        return write_typed_array<std::uint16_t>(group, path, arr);
    if (kind == 'u' && itemsize == 4)
        return write_typed_array<std::uint32_t>(group, path, arr);
    if (kind == 'u' && itemsize == 8)
        return write_typed_array<std::uint64_t>(group, path, arr);
    if (kind == 'b' || (kind == 'i' && dt.attr("name").cast<std::string>() == "bool"))
        return write_typed_array<std::uint8_t>(group, path, arr);

    // Fallback: let h5py encode uncommon dtypes, then we still created nothing.
    throw Hdf5ExportError("unsupported numpy dtype for HDF5 dataset: " +
                          py::str(py::object(dt)).cast<std::string>());
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
write_vlen_bytes(hid_t loc, std::string const& path, py::bytes value)
{
    char* buf = nullptr;
    py::ssize_t len = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(value.ptr(), &buf, &len) != 0)
        throw Hdf5ExportError("invalid bytes object");
    std::string s(buf, static_cast<size_t>(len));
    write_vlen_string(loc, path, s, false);
}

template<typename T>
void
write_scalar(HighFive::Group& group, std::string const& path, T const& value)
{
    auto space = HighFive::DataSpace(HighFive::DataSpace::DataspaceType::dataspace_scalar);
    auto dset = group.createDataSet(path, space, HighFive::create_datatype<T>());
    dset.write_raw(&value);
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

} // namespace

hid_t
hid_from_h5py(py::handle obj)
{
    return obj.attr("id").attr("id").cast<hid_t>();
}

bool
is_h5py_file(py::handle obj)
{
    return py::isinstance(obj, h5py_mod().attr("File"));
}

bool
is_h5py_group(py::handle obj)
{
    auto h5py = h5py_mod();
    return py::isinstance(obj, h5py.attr("Group")) || py::isinstance(obj, h5py.attr("File"));
}

bool
is_h5py_dataset(py::handle obj)
{
    return py::isinstance(obj, h5py_mod().attr("Dataset"));
}

bool
is_h5py_object(py::handle obj)
{
    return is_h5py_group(obj) || is_h5py_dataset(obj);
}

HighFive::Group
wrap_group(py::handle obj)
{
    if (is_h5py_file(obj)) {
        hid_t fid = hid_from_h5py(obj);
        hid_t gid = H5Gopen2(fid, "/", H5P_DEFAULT);
        if (gid < 0)
            throw Hdf5ExportError("failed to open root group");
        return HighFive::detail::make_group(gid);
    }
    hid_t hid = hid_from_h5py(obj);
    check_hdf5(H5Iinc_ref(hid), "H5Iinc_ref");
    return HighFive::detail::make_group(hid);
}

py::object
h5py_from_hid(hid_t hid)
{
    auto h5py = h5py_mod();
    H5I_type_t t = H5Iget_type(hid);
    check_hdf5(H5Iinc_ref(hid), "H5Iinc_ref");
    if (t == H5I_GROUP) {
        py::object gid = h5py.attr("h5g").attr("GroupID")(hid);
        return h5py.attr("Group")(gid);
    }
    if (t == H5I_DATASET) {
        py::object did = h5py.attr("h5d").attr("DatasetID")(hid);
        return h5py.attr("Dataset")(did);
    }
    if (t == H5I_FILE) {
        hid_t gid = H5Gopen2(hid, "/", H5P_DEFAULT);
        py::object gpy = h5py.attr("h5g").attr("GroupID")(gid);
        return h5py.attr("Group")(gpy);
    }
    H5Idec_ref(hid);
    throw Hdf5ExportError("cannot wrap HDF5 id as h5py object");
}

py::object
h5py_getitem(py::handle parent, std::string const& path)
{
    return parent[py::str(path)];
}

py::object
h5_create_group(py::handle parent, std::string const& path)
{
    auto group = wrap_group(parent);
    std::string p = path;
    if (p.size() > 1 && p.front() == '/')
        p = p.substr(1);
    group.createGroup(p);
    return h5py_getitem(parent, path);
}

void
h5_hard_link(py::handle parent, std::string const& path, py::handle src)
{
    hid_t dest = hid_from_h5py(parent);
    hid_t src_id = hid_from_h5py(src);
    std::string p = path;
    herr_t st = H5Lcreate_hard(src_id, ".", dest, p.c_str(), H5P_DEFAULT, H5P_DEFAULT);
    check_hdf5(st, "H5Lcreate_hard");
}

void
h5_write_dataset(py::handle parent, std::string const& path, py::handle obj)
{
    auto group = wrap_group(parent);
    std::string p = path;
    if (p.size() > 1 && p.front() == '/')
        p = p.substr(1);

    py::module_ np = py::module_::import("numpy");
    if (py::isinstance(obj, np.attr("ndarray")) &&
        !py::isinstance(obj, np.attr("ma").attr("MaskedArray"))) {
        write_numpy_array(group, p, py::reinterpret_borrow<py::array>(obj));
        return;
    }
    if (py::isinstance<py::str>(obj)) {
        write_vlen_string(group.getId(), p, obj.cast<std::string>(), true);
        return;
    }
    if (py::isinstance<py::bytes>(obj)) {
        write_vlen_bytes(group.getId(), p, py::reinterpret_borrow<py::bytes>(obj));
        return;
    }
    if (obj.ptr() == Py_True || obj.ptr() == Py_False) {
        std::uint8_t v = PyObject_IsTrue(obj.ptr()) ? 1 : 0;
        write_scalar<std::uint8_t>(group, p, v);
        return;
    }
    if (py::isinstance<py::int_>(obj) && !py::isinstance(obj, np.attr("integer"))) {
        py::int_ pyi = py::reinterpret_borrow<py::int_>(obj);
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
        if (overflow == 0 && !PyErr_Occurred()) {
            write_scalar<std::int64_t>(group, p, static_cast<std::int64_t>(v));
            return;
        }
        PyErr_Clear();
        unsigned long long uv = PyLong_AsUnsignedLongLong(obj.ptr());
        if (!PyErr_Occurred()) {
            write_scalar<std::uint64_t>(group, p, static_cast<std::uint64_t>(uv));
            return;
        }
        PyErr_Clear();
        throw py::type_error("No conversion path for dtype: dtype('O') "
                             "and no native HDF5 equivalent");
    }
    if (py::isinstance<py::float_>(obj)) {
        write_scalar<double>(group, p, obj.cast<double>());
        return;
    }
    if (PyComplex_Check(obj.ptr())) {
        std::complex<double> z(PyComplex_RealAsDouble(obj.ptr()),
                               PyComplex_ImagAsDouble(obj.ptr()));
        write_scalar<std::complex<double>>(group, p, z);
        return;
    }
    if (py::isinstance(obj, np.attr("generic"))) {
        py::array arr = np.attr("asarray")(obj);
        write_numpy_array(group, p, arr);
        return;
    }
    throw Hdf5ExportError("don't know how to write dataset for object");
}

void
h5_set_item(py::handle parent, std::string const& path, py::handle obj)
{
    if (is_h5py_object(obj)) {
        h5_hard_link(parent, path, obj);
        return;
    }
    h5_write_dataset(parent, path, obj);
}

void
h5_set_attr(py::handle h5obj, std::string const& name, py::handle value)
{
    h5obj.attr("attrs")[py::str(name)] = value;
}

py::object
h5_get_attr(py::handle h5obj, std::string const& name)
{
    py::object res = h5obj.attr("attrs").attr("get")(py::str(name));
    if (res.is_none())
        return py::none();
    if (py::isinstance<py::bytes>(res))
        return py::str(res.cast<std::string>());
    return res;
}

bool
h5_contains(py::handle parent, std::string const& name)
{
    return py::bool_(parent.attr("__contains__")(name));
}

} // namespace hdf5_io
