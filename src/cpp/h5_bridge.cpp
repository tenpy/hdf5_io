#include <tenpy_hdf5/exceptions.h>
#include <tenpy_hdf5/h5_bridge.h>
#include <tenpy_hdf5/h5_ops.h>

#include <hdf5.h>
#include <highfive/highfive.hpp>

#include <complex>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tenpy_hdf5 {

namespace {

py::module_
h5py_mod()
{
    return py::module_::import_("h5py");
}

HighFive::DataSpace
numpy_space(NumpyArray const& arr)
{
    if (arr.ndim() == 0)
        return HighFive::DataSpace(HighFive::DataSpace::DataspaceType::dataspace_scalar);
    std::vector<size_t> dims(static_cast<size_t>(arr.ndim()));
    for (Py_ssize_t i = 0; i < arr.ndim(); ++i)
        dims[static_cast<size_t>(i)] = static_cast<size_t>(arr.shape(i));
    return HighFive::DataSpace(dims);
}

template<typename T>
void
write_typed_array(HighFive::Group& group, std::string const& path, NumpyArray arr)
{
    arr = arr.as_c_contiguous();
    auto space = numpy_space(arr);
    auto dset = group.createDataSet(path, space, HighFive::create_datatype<T>());
    if (arr.size() > 0 || arr.ndim() == 0)
        dset.write_raw(static_cast<T const*>(arr.data()));
}

void
write_numpy_array(HighFive::Group& group, std::string const& path, NumpyArray arr)
{
    NumpyDtype dt = arr.dtype();
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
    if (kind == 'b' || (kind == 'i' && dt.name() == "bool"))
        return write_typed_array<std::uint8_t>(group, path, arr);

    throw Hdf5ExportError("unsupported numpy dtype for HDF5 dataset: " +
                          py::cast<std::string>(py::str(dt)));
}

void
write_vlen_bytes(hid_t loc, std::string const& path, py::bytes value)
{
    char* buf = nullptr;
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(value.ptr(), &buf, &len) != 0) {
        PyErr_Clear();
        throw Hdf5ExportError("invalid bytes object");
    }
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
    return py::cast<hid_t>(obj.attr("id").attr("id"));
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
        if (gid < 0) {
            throw Hdf5ExportError(
              "failed to open root group (invalid HDF5 identifier). "
              "tenpy_hdf5 and h5py must link the same libhdf5; pip h5py wheels "
              "vendor a separate HDF5. Prefer conda-forge hdf5+h5py, or "
              "`pip install --no-binary=h5py h5py` against the libhdf5 used to "
              "build tenpy-hdf5.");
        }
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
    return parent[to_str(path)];
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
h5_write_dataset(HighFive::Group& parent, std::string const& path, NumpyArray const& obj)
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
    py::module_ np = py::module_::import_("numpy");
    if (py::isinstance(obj, np.attr("ndarray")) &&
        !py::isinstance(obj, np.attr("ma").attr("MaskedArray"))) {
        h5_write_dataset(group, path, py::borrow<NumpyArray>(obj));
        return;
    }
    if (py::isinstance<py::str>(obj)) {
        h5_write_dataset(group, path, py::cast<std::string>(obj));
        return;
    }
    if (py::isinstance<py::bytes>(obj)) {
        h5_write_dataset(group, path, py::borrow<py::bytes>(obj));
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
        h5_write_dataset(group, path, py::cast<double>(obj));
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
          group, path, NumpyArray::from_object(py::module_::import_("numpy").attr("asarray")(obj)));
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
    h5obj.attr("attrs")[to_str(name)] = value;
}

py::object
h5_get_attr(py::handle h5obj, std::string const& name)
{
    py::object res = h5obj.attr("attrs").attr("get")(to_str(name));
    if (res.is_none())
        return py::none();
    if (py::isinstance<py::bytes>(res))
        return to_str(py::cast<std::string>(res));
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
        h5_set_attr(loc, name, py::cast<std::string>(value));
        return;
    }
    if (value.ptr() == Py_True || value.ptr() == Py_False) {
        h5_set_attr(loc, name, PyObject_IsTrue(value.ptr()) != 0);
        return;
    }
    py::module_ np = py::module_::import_("numpy");
    if (py::isinstance<py::int_>(value) && !py::isinstance(value, np.attr("integer"))) {
        h5_set_attr(loc, name, py::cast<std::int64_t>(value));
        return;
    }
    py::object pyobj = h5py_from_hid(loc);
    pyobj.attr("attrs")[to_str(name)] = value;
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
            return to_str(*s);
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
    return pyobj.attr("attrs").attr("get")(to_str(name));
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

NumpyDtype
dtype_from_hdf5(hid_t ntype)
{
    H5T_class_t cls = H5Tget_class(ntype);
    size_t sz = H5Tget_size(ntype);
    if (cls == H5T_ENUM) {
        if (is_hdf5_bool_enum(ntype))
            return NumpyDtype::from_spec("bool");
        hid_t super = H5Tget_super(ntype);
        NumpyDtype dt = dtype_from_hdf5(super);
        H5Tclose(super);
        return dt;
    }
    if (cls == H5T_BITFIELD) {
        if (sz == 1)
            return NumpyDtype::from_spec("uint8");
        if (sz == 2)
            return NumpyDtype::from_spec("uint16");
        if (sz == 4)
            return NumpyDtype::from_spec("uint32");
        if (sz == 8)
            return NumpyDtype::from_spec("uint64");
    }
    if (cls == H5T_INTEGER) {
        H5T_sign_t sign = H5Tget_sign(ntype);
        if (sign == H5T_SGN_NONE) {
            if (sz == 1)
                return NumpyDtype::from_spec("uint8");
            if (sz == 2)
                return NumpyDtype::from_spec("uint16");
            if (sz == 4)
                return NumpyDtype::from_spec("uint32");
            if (sz == 8)
                return NumpyDtype::from_spec("uint64");
        } else {
            if (sz == 1)
                return NumpyDtype::from_spec("int8");
            if (sz == 2)
                return NumpyDtype::from_spec("int16");
            if (sz == 4)
                return NumpyDtype::from_spec("int32");
            if (sz == 8)
                return NumpyDtype::from_spec("int64");
        }
    } else if (cls == H5T_FLOAT) {
        if (sz == 2)
            return NumpyDtype::from_spec("float16");
        if (sz == 4)
            return NumpyDtype::from_spec("float32");
        if (sz == 8)
            return NumpyDtype::from_spec("float64");
    } else if (cls == H5T_COMPOUND) {
        if (sz == sizeof(std::complex<float>))
            return NumpyDtype::from_spec("complex64");
        if (sz == sizeof(std::complex<double>))
            return NumpyDtype::from_spec("complex128");
    }
    throw Hdf5ExportError("unsupported HDF5 dataset dtype");
}

NumpyArray
h5_read_array_h5py(hid_t dset)
{
    py::object ds = h5py_from_hid(dset);
    py::object val = ds.attr("__getitem__")(py::ellipsis());
    return NumpyArray::from_object(py::module_::import_("numpy").attr("asarray")(val));
}

} // namespace

NumpyArray
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
        NumpyArray arr;
        if (space_type == H5S_SCALAR) {
            NumpyDtype dt = dtype_from_hdf5(ntype);
            arr = NumpyArray::empty(dt, std::vector<Py_ssize_t>{});
            check_hdf5(H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr.mutable_data()),
                       "H5Dread scalar");
        } else {
            int ndims = H5Sget_simple_extent_ndims(space);
            std::vector<hsize_t> dims(static_cast<size_t>(ndims));
            H5Sget_simple_extent_dims(space, dims.data(), nullptr);
            std::vector<Py_ssize_t> shape(dims.begin(), dims.end());
            NumpyDtype dt = dtype_from_hdf5(ntype);
            arr = NumpyArray::empty(dt, shape);
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

} // namespace tenpy_hdf5
