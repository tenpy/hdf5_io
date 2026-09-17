#include <hdf5_io/exceptions.h>
#include <hdf5_io/h5_bridge.h>
#include <hdf5_io/h5_ops.h>

#include <hdf5.h>
#include <highfive/highfive.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hdf5_io {

namespace {

py::module_
h5py_mod()
{
    return py::module_::import("h5py");
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

    throw Hdf5ExportError("unsupported numpy dtype for HDF5 dataset: " +
                          py::str(py::object(dt)).cast<std::string>());
}

void
write_vlen_bytes(hid_t loc, std::string const& path, py::bytes value)
{
    char* buf = nullptr;
    py::ssize_t len = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(value.ptr(), &buf, &len) != 0)
        throw Hdf5ExportError("invalid bytes object");
    std::string s(buf, static_cast<size_t>(len));
    // ASCII vlen string dataset (same as former bridge)
    hid_t type = H5Tcopy(H5T_C_S1);
    check_hdf5(H5Tset_size(type, H5T_VARIABLE), "H5Tset_size");
    check_hdf5(H5Tset_cset(type, H5T_CSET_ASCII), "H5Tset_cset");
    check_hdf5(H5Tset_strpad(type, H5T_STR_NULLTERM), "H5Tset_strpad");
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dset = H5Dcreate2(loc, path.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        H5Sclose(space);
        H5Tclose(type);
        throw Hdf5ExportError("failed to create string dataset " + path);
    }
    char const* ptr = s.c_str();
    herr_t st = H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ptr);
    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(type);
    check_hdf5(st, "H5Dwrite string");
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

std::string
object_token(py::handle obj)
{
    return object_token(hid_from_h5py(obj));
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
    h5_create_group(group, path);
    return h5py_getitem(parent, path);
}

void
h5_hard_link(py::handle parent, std::string const& path, py::handle src)
{
    auto group = wrap_group(parent);
    h5_hard_link(group, path, hid_from_h5py(src));
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, py::array const& obj)
{
    write_numpy_array(parent, normalized_path(path), obj);
}

void
h5_write_dataset(HighFive::Group& parent, std::string const& path, py::bytes const& obj)
{
    write_vlen_bytes(parent.getId(), normalized_path(path), obj);
}

void
h5_write_dataset(py::handle parent, std::string const& path, py::handle obj)
{
    auto group = wrap_group(parent);
    py::module_ np = py::module_::import("numpy");
    if (py::isinstance(obj, np.attr("ndarray")) &&
        !py::isinstance(obj, np.attr("ma").attr("MaskedArray"))) {
        h5_write_dataset(group, path, py::reinterpret_borrow<py::array>(obj));
        return;
    }
    if (py::isinstance<py::str>(obj)) {
        h5_write_dataset(group, path, obj.cast<std::string>());
        return;
    }
    if (py::isinstance<py::bytes>(obj)) {
        h5_write_dataset(group, path, py::reinterpret_borrow<py::bytes>(obj));
        return;
    }
    if (obj.ptr() == Py_True || obj.ptr() == Py_False) {
        h5_write_dataset(group, path, PyObject_IsTrue(obj.ptr()) != 0);
        return;
    }
    if (py::isinstance<py::int_>(obj) && !py::isinstance(obj, np.attr("integer"))) {
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
        if (overflow == 0 && !PyErr_Occurred()) {
            h5_write_dataset(group, path, static_cast<std::int64_t>(v));
            return;
        }
        PyErr_Clear();
        unsigned long long uv = PyLong_AsUnsignedLongLong(obj.ptr());
        if (!PyErr_Occurred()) {
            h5_write_dataset(group, path, static_cast<std::uint64_t>(uv));
            return;
        }
        PyErr_Clear();
        throw py::type_error("No conversion path for dtype: dtype('O') "
                             "and no native HDF5 equivalent");
    }
    if (py::isinstance<py::float_>(obj)) {
        h5_write_dataset(group, path, obj.cast<double>());
        return;
    }
    if (PyComplex_Check(obj.ptr())) {
        h5_write_dataset(group,
                         path,
                         std::complex<double>(PyComplex_RealAsDouble(obj.ptr()),
                                              PyComplex_ImagAsDouble(obj.ptr())));
        return;
    }
    if (py::isinstance(obj, np.attr("generic"))) {
        h5_write_dataset(
          group, path, py::module_::import("numpy").attr("asarray")(obj).cast<py::array>());
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
    auto group = wrap_group(parent);
    return h5_contains(group, name);
}

void
h5_set_attr(hid_t loc, std::string const& name, py::handle value)
{
    if (py::isinstance<py::str>(value)) {
        h5_set_attr(loc, name, value.cast<std::string>());
        return;
    }
    if (value.ptr() == Py_True || value.ptr() == Py_False) {
        h5_set_attr(loc, name, PyObject_IsTrue(value.ptr()) != 0);
        return;
    }
    py::module_ np = py::module_::import("numpy");
    if (py::isinstance<py::int_>(value) && !py::isinstance(value, np.attr("integer"))) {
        h5_set_attr(loc, name, value.cast<std::int64_t>());
        return;
    }
    py::object pyobj = h5py_from_hid(loc);
    pyobj.attr("attrs")[py::str(name)] = value;
}

namespace {

bool
attr_dataspace_is_scalar(hid_t loc, std::string const& name)
{
    if (H5Aexists(loc, name.c_str()) <= 0)
        return false;
    hid_t attr = H5Aopen(loc, name.c_str(), H5P_DEFAULT);
    if (attr < 0)
        return false;
    hid_t space = H5Aget_space(attr);
    bool scalar = space >= 0 && H5Sget_simple_extent_type(space) == H5S_SCALAR;
    if (space >= 0)
        H5Sclose(space);
    H5Aclose(attr);
    return scalar;
}

} // namespace

py::object
h5_get_attr(hid_t loc, std::string const& name)
{
    if (H5Aexists(loc, name.c_str()) <= 0)
        return py::none();

    // Scalar fast-path only. Array attrs (e.g. string labels) must go through h5py;
    // reading them with H5Aread into a single value returns only the first element.
    if (attr_dataspace_is_scalar(loc, name)) {
        if (auto s = h5_get_attr_string(loc, name))
            return py::str(*s);
        if (auto v = h5_get_attr_int64(loc, name))
            return py::int_(*v);
        hid_t attr = H5Aopen(loc, name.c_str(), H5P_DEFAULT);
        if (attr < 0)
            throw Hdf5ExportError("failed to open attribute " + name);
        hid_t type = H5Aget_type(attr);
        H5T_class_t cls = H5Tget_class(type);
        if (cls == H5T_FLOAT) {
            double v = 0;
            hid_t ntype = H5Tcopy(H5T_NATIVE_DOUBLE);
            check_hdf5(H5Aread(attr, ntype, &v), "H5Aread float");
            H5Tclose(ntype);
            H5Tclose(type);
            H5Aclose(attr);
            return py::float_(v);
        }
        H5Tclose(type);
        H5Aclose(attr);
    }

    py::object pyobj = h5py_from_hid(loc);
    return pyobj.attr("attrs").attr("get")(py::str(name));
}

namespace {

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

py::dtype
dtype_from_hdf5(hid_t ntype)
{
    H5T_class_t cls = H5Tget_class(ntype);
    size_t sz = H5Tget_size(ntype);
    if (cls == H5T_ENUM) {
        if (is_hdf5_bool_enum(ntype))
            return py::dtype("bool");
        hid_t super = H5Tget_super(ntype);
        py::dtype dt = dtype_from_hdf5(super);
        H5Tclose(super);
        return dt;
    }
    if (cls == H5T_BITFIELD) {
        if (sz == 1)
            return py::dtype::of<std::uint8_t>();
        if (sz == 2)
            return py::dtype::of<std::uint16_t>();
        if (sz == 4)
            return py::dtype::of<std::uint32_t>();
        if (sz == 8)
            return py::dtype::of<std::uint64_t>();
    }
    if (cls == H5T_INTEGER) {
        H5T_sign_t sign = H5Tget_sign(ntype);
        if (sign == H5T_SGN_NONE) {
            if (sz == 1)
                return py::dtype::of<std::uint8_t>();
            if (sz == 2)
                return py::dtype::of<std::uint16_t>();
            if (sz == 4)
                return py::dtype::of<std::uint32_t>();
            if (sz == 8)
                return py::dtype::of<std::uint64_t>();
        } else {
            if (sz == 1)
                return py::dtype::of<std::int8_t>();
            if (sz == 2)
                return py::dtype::of<std::int16_t>();
            if (sz == 4)
                return py::dtype::of<std::int32_t>();
            if (sz == 8)
                return py::dtype::of<std::int64_t>();
        }
    } else if (cls == H5T_FLOAT) {
        if (sz == 2)
            return py::dtype("float16");
        if (sz == 4)
            return py::dtype::of<float>();
        if (sz == 8)
            return py::dtype::of<double>();
    } else if (cls == H5T_COMPOUND) {
        if (sz == sizeof(std::complex<float>))
            return py::dtype::of<std::complex<float>>();
        if (sz == sizeof(std::complex<double>))
            return py::dtype::of<std::complex<double>>();
    }
    throw Hdf5ExportError("unsupported HDF5 dataset dtype");
}

py::array
h5_read_array_h5py(hid_t dset)
{
    py::object ds = h5py_from_hid(dset);
    py::object val = ds.attr("__getitem__")(py::ellipsis());
    return py::module_::import("numpy").attr("asarray")(val).cast<py::array>();
}

} // namespace

py::array
h5_read_array(hid_t dset)
{
    hid_t ftype = H5Dget_type(dset);
    hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
    hid_t space = H5Dget_space(dset);
    auto close_types = [&]() {
        H5Sclose(space);
        H5Tclose(ntype);
        H5Tclose(ftype);
    };
    try {
        H5S_class_t space_type = H5Sget_simple_extent_type(space);
        py::array arr;
        if (space_type == H5S_SCALAR) {
            py::dtype dt = dtype_from_hdf5(ntype);
            arr = py::array(dt, std::vector<py::ssize_t>{});
            check_hdf5(H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr.mutable_data()),
                       "H5Dread scalar");
        } else {
            int ndims = H5Sget_simple_extent_ndims(space);
            std::vector<hsize_t> dims(static_cast<size_t>(ndims));
            H5Sget_simple_extent_dims(space, dims.data(), nullptr);
            std::vector<py::ssize_t> shape(dims.begin(), dims.end());
            py::dtype dt = dtype_from_hdf5(ntype);
            arr = py::array(dt, shape);
            if (arr.size() > 0)
                check_hdf5(H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr.mutable_data()),
                           "H5Dread array");
        }
        close_types();
        return arr;
    } catch (Hdf5ExportError const&) {
        close_types();
        return h5_read_array_h5py(dset);
    }
}

} // namespace hdf5_io
