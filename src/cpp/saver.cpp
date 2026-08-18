#include <hdf5_io/h5_bridge.h>
#include <hdf5_io/hdf5_io.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <utility>

namespace hdf5_io {

namespace {

std::uint64_t
py_id(py::handle obj)
{
    return py::module_::import("builtins").attr("id")(obj).cast<std::uint64_t>();
}

py::object
self_obj(Hdf5Saver* self)
{
    return py::cast(self, py::return_value_policy::reference);
}

std::string
ensure_subpath(std::string const& path)
{
    return (!path.empty() && path.back() == '/') ? path : (path + "/");
}

} // namespace

Hdf5Saver::Hdf5Saver(py::object h5group_, py::object format_selection_)
  : h5group(std::move(h5group_))
  , native_group(wrap_group(h5group))
{
    if (format_selection_.is_none())
        format_selection = py::dict();
    else
        format_selection = py::reinterpret_borrow<py::dict>(format_selection_);
}

Hdf5Saver::~Hdf5Saver()
{
    for (auto const& [_, entry] : memo_save_ids) {
        if (entry.first >= 0)
            H5Idec_ref(entry.first);
    }
}

py::dict
Hdf5Saver::memo_save() const
{
    py::dict out;
    for (auto const& [obj_id, entry] : memo_save_ids)
        out[py::int_(obj_id)] = py::make_tuple(h5py_from_hid(entry.first), entry.second);
    return out;
}

py::dict
Hdf5Saver::dispatch_save() const
{
    return py::type::of<Hdf5Saver>().attr("dispatch_save");
}

py::object
Hdf5Saver::save(py::object obj, std::string path)
{
    hid_t id = save_hid(std::move(obj), std::move(path));
    if (id < 0)
        return py::none();
    py::object pygr = h5py_from_hid(id);
    H5Idec_ref(id);
    return pygr;
}

hid_t
Hdf5Saver::save_hid(py::object obj, std::string path)
{
    std::uint64_t obj_id = py_id(obj);
    auto in_memo = memo_save_ids.find(obj_id);
    if (in_memo != memo_save_ids.end()) {
        h5_set_item(native_group, path, in_memo->second.first);
        return h5_open(native_group, path);
    }

    auto const& cpp_disp = cpp_save_dispatch();
    auto cpp_it = cpp_disp.find(Py_TYPE(obj.ptr()));
    if (cpp_it != cpp_disp.end())
        return (this->*(cpp_it->second.first))(obj, path, cpp_it->second.second);

    py::object disp = dispatch_save().attr("get")(py::type::of(obj));
    if (!disp.is_none()) {
        py::object f = disp[py::int_(0)];
        py::object type_repr = disp[py::int_(1)];
        py::object pygr = f(self_obj(this), obj, path, type_repr);
        if (pygr.is_none())
            return H5I_INVALID_HID;
        hid_t id = hid_from_h5py(pygr);
        H5Iinc_ref(id);
        return id;
    }

    py::object obj_save_hdf5 = py::getattr(obj, "save_hdf5", py::none());
    if (!obj_save_hdf5.is_none()) {
        auto [group, subpath] = create_group_for_obj(path, obj);
        h5_set_attr(group.getId(), ATTR_TYPE, std::string(REPR_HDF5EXPORTABLE));
        h5_set_attr(group.getId(), ATTR_CLASS, obj.attr("__class__").attr("__qualname__").cast<std::string>());
        h5_set_attr(group.getId(), ATTR_MODULE, obj.attr("__class__").attr("__module__").cast<std::string>());
        py::object h5gr = h5py_from_hid(group.getId());
        obj_save_hdf5(self_obj(this), h5gr, subpath);
        hid_t id = group.getId();
        H5Iinc_ref(id);
        return id;
    }

    py::module_::import("warnings")
      .attr("warn")(py::str("Hdf5Saver: object of type {0!r} without explicit HDF5 format; "
                            "fall back to pickle protocol")
                      .attr("format")(py::type::of(obj)),
                    py::module_::import("builtins").attr("UserWarning"));

    py::object obj_reduce = py::getattr(obj, "__reduce__", py::none());
    if (!obj_reduce.is_none()) {
        py::object rv = obj_reduce();
        if (py::isinstance<py::str>(rv))
            return save_global(obj, path, REPR_GLOBAL);
        if (!py::isinstance<py::tuple>(rv))
            throw Hdf5ExportError("Wrong return value of reduce");
        py::tuple tup = py::reinterpret_borrow<py::tuple>(rv);
        if (tup.size() < 2 || tup.size() >= 7)
            throw Hdf5ExportError("Wrong return value of reduce");
        py::object func = tup[0];
        py::object args = tup[1];
        py::object state = tup.size() > 2 ? tup[2] : py::none();
        py::object listitems = tup.size() > 3 ? tup[3] : py::none();
        py::object dictitems = tup.size() > 4 ? tup[4] : py::none();
        py::object state_setter = tup.size() > 5 ? tup[5] : py::none();
        py::object pygr = save_reduce(
          func, args, state, listitems, dictitems, state_setter, obj, py::str(path));
        hid_t id = hid_from_h5py(pygr);
        H5Iinc_ref(id);
        return id;
    }

    throw Hdf5ExportError("Don't know how to save object of type " +
                          py::repr(py::type::of(obj)).cast<std::string>());
}

std::pair<HighFive::Group, std::string>
Hdf5Saver::create_group_for_obj(std::string const& path, py::object obj)
{
    HighFive::Group gr;
    if (path == "/") {
        hid_t id = native_group.getId();
        H5Iinc_ref(id);
        gr = group_from_hid(id);
    } else {
        gr = h5_create_group(native_group, path);
    }
    std::string subpath = ensure_subpath(path);
    memorize_save(gr.getId(), obj);
    return { std::move(gr), subpath };
}

void
Hdf5Saver::memorize_save(hid_t h5obj, py::object obj)
{
    std::uint64_t obj_id = py_id(obj);
    if (memo_save_ids.contains(obj_id))
        throw Hdf5ExportError("object already memorized");
    H5Iinc_ref(h5obj);
    memo_save_ids.emplace(obj_id, std::make_pair(h5obj, obj));
}

void
Hdf5Saver::memorize_save(py::object h5gr, py::object obj)
{
    memorize_save(hid_from_h5py(h5gr), std::move(obj));
}

py::object
Hdf5Saver::save_reduce(py::object func,
                       py::object args,
                       py::object state,
                       py::object listitems,
                       py::object dictitems,
                       py::object state_setter,
                       py::object obj,
                       py::object path)
{
    std::string p = path.is_none() ? std::string("/") : path.cast<std::string>();
    auto [group, subpath] = create_group_for_obj(p, obj);
    h5_set_attr(group.getId(), ATTR_TYPE, std::string(REPR_REDUCE));
    save_hid(func, subpath + "func");
    save_hid(args, subpath + "args");
    if (!state.is_none())
        save_hid(state, subpath + "state");
    if (!listitems.is_none())
        save_hid(state, subpath + "listitems");
    if (!dictitems.is_none())
        save_hid(state, subpath + "dictitems");
    if (!state_setter.is_none())
        save_hid(state, subpath + "state_setter");
    return h5py_from_hid(group.getId());
}

hid_t
Hdf5Saver::save_none(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    h5_write_dataset(native_group, path, std::string(REPR_NONE));
    hid_t id = h5_open(native_group, path);
    h5_set_attr(id, ATTR_TYPE, std::string(REPR_NONE));
    memorize_save(id, obj);
    return id;
}

hid_t
Hdf5Saver::save_dataset(py::object obj, std::string const& path, std::string const& type_repr)
{
    std::string repr = type_repr;
    py::module_ np = py::module_::import("numpy");
    auto write_int = [&]() {
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
        if (overflow == 0 && !PyErr_Occurred()) {
            h5_write_dataset(native_group, path, static_cast<std::int64_t>(v));
            return;
        }
        PyErr_Clear();
        unsigned long long uv = PyLong_AsUnsignedLongLong(obj.ptr());
        if (PyErr_Occurred()) {
            PyErr_Clear();
            throw py::type_error(
              "No conversion path for dtype: dtype('O') and no native HDF5 equivalent");
        }
        h5_write_dataset(native_group, path, static_cast<std::uint64_t>(uv));
    };

    try {
        if (repr == REPR_ARRAY)
            h5_write_dataset(native_group, path, py::reinterpret_borrow<py::array>(obj));
        else if (repr == REPR_STR)
            h5_write_dataset(native_group, path, obj.cast<std::string>());
        else if (repr == REPR_BYTES)
            h5_write_dataset(native_group, path, py::reinterpret_borrow<py::bytes>(obj));
        else if (repr == REPR_BOOL)
            h5_write_dataset(native_group, path, PyObject_IsTrue(obj.ptr()) != 0);
        else if (repr == REPR_INT)
            write_int();
        else if (repr == REPR_FLOAT)
            h5_write_dataset(native_group, path, obj.cast<double>());
        else if (repr == REPR_COMPLEX)
            h5_write_dataset(native_group,
                             path,
                             std::complex<double>(PyComplex_RealAsDouble(obj.ptr()),
                                                  PyComplex_ImagAsDouble(obj.ptr())));
        else
            h5_write_dataset(native_group, path, np.attr("asarray")(obj).cast<py::array>());
    } catch (py::error_already_set& e) {
        if (repr != REPR_INT || !e.matches(PyExc_TypeError))
            throw;
        std::string msg = e.what();
        if (msg.find("no native HDF5 equivalent") == std::string::npos)
            throw;
        repr = REPR_INT_AS_STR;
        h5_write_dataset(native_group, path, obj.attr("__str__")().cast<std::string>());
    } catch (Hdf5ExportError const& e) {
        if (repr != REPR_INT ||
            std::string(e.what()).find("no native HDF5 equivalent") == std::string::npos)
            throw;
        repr = REPR_INT_AS_STR;
        h5_write_dataset(native_group, path, obj.attr("__str__")().cast<std::string>());
    } catch (py::type_error const& e) {
        if (repr != REPR_INT ||
            std::string(e.what()).find("no native HDF5 equivalent") == std::string::npos)
            throw;
        repr = REPR_INT_AS_STR;
        h5_write_dataset(native_group, path, obj.attr("__str__")().cast<std::string>());
    }
    hid_t id = h5_open(native_group, path);
    h5_set_attr(id, ATTR_TYPE, repr);
    memorize_save(id, obj);
    return id;
}

hid_t
Hdf5Saver::save_masked_array(py::object obj, std::string const& path, std::string const& type_repr)
{
    py::object filled = obj.attr("filled")();
    py::object fill_value = obj.attr("fill_value");
    py::object np = py::module_::import("numpy");
    py::object mask = obj.attr("mask");
    py::object cmp = filled.attr("__eq__")(fill_value);
    py::object mask_cmp = cmp.attr("__eq__")(mask);
    hid_t id;
    if (np.attr("any")(mask_cmp).cast<bool>()) {
        auto created = create_group_for_obj(path, obj);
        h5_write_dataset(created.first, "data", obj.attr("data").cast<py::array>());
        h5_write_dataset(created.first, "mask", obj.attr("mask").cast<py::array>());
        h5_set_attr(created.first.getId(), "saved_mask", true);
        id = created.first.getId();
        H5Iinc_ref(id);
        h5_set_attr(id, ATTR_TYPE, type_repr);
        h5_set_attr(id, "fill_value", fill_value);
        return id;
    }
    h5_write_dataset(native_group, path, filled.cast<py::array>());
    id = h5_open(native_group, path);
    h5_set_attr(id, "saved_mask", false);
    memorize_save(id, obj);
    h5_set_attr(id, ATTR_TYPE, type_repr);
    h5_set_attr(id, "fill_value", fill_value);
    return id;
}

hid_t
Hdf5Saver::save_iterable(py::object obj, std::string const& path, std::string const& type_repr)
{
    auto [group, subpath] = create_group_for_obj(path, obj);
    h5_set_attr(group.getId(), ATTR_TYPE, type_repr);
    save_iterable_content(obj, group, subpath);
    hid_t id = group.getId();
    H5Iinc_ref(id);
    return id;
}

void
Hdf5Saver::save_iterable_content(py::object obj, HighFive::Group& h5gr, std::string const& subpath)
{
    py::ssize_t n = py::len(obj);
    h5_set_attr(h5gr.getId(), ATTR_LEN, static_cast<std::int64_t>(n));
    py::ssize_t i = 0;
    for (auto elem : obj) {
        hid_t child = save_hid(py::reinterpret_borrow<py::object>(elem), subpath + std::to_string(i));
        if (child >= 0)
            H5Idec_ref(child);
        ++i;
    }
}

hid_t
Hdf5Saver::save_dict(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    auto [group, subpath] = create_group_for_obj(path, obj);
    std::string tr = save_dict_content(obj, group, subpath);
    h5_set_attr(group.getId(), ATTR_TYPE, tr);
    hid_t id = group.getId();
    H5Iinc_ref(id);
    return id;
}

std::string
Hdf5Saver::save_dict_content(py::object obj, HighFive::Group& h5gr, std::string const& subpath)
{
    (void)h5gr;
    bool simple_keys = true;
    for (auto k : obj.attr("keys")()) {
        if (!valid_hdf5_path_component(k)) {
            simple_keys = false;
            break;
        }
    }
    if (simple_keys) {
        for (auto item : obj.attr("items")()) {
            py::tuple kv = py::reinterpret_borrow<py::tuple>(item);
            hid_t child = save_hid(kv[1], subpath + kv[0].cast<std::string>());
            if (child >= 0)
                H5Idec_ref(child);
        }
        return REPR_DICT_SIMPLE;
    }
    hid_t keys = save_iterable(obj.attr("keys")(), subpath + "keys", REPR_LIST);
    hid_t values = save_iterable(obj.attr("values")(), subpath + "values", REPR_LIST);
    if (keys >= 0)
        H5Idec_ref(keys);
    if (values >= 0)
        H5Idec_ref(values);
    return REPR_DICT_GENERAL;
}

hid_t
Hdf5Saver::save_range(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    auto [group, subpath] = create_group_for_obj(path, obj);
    h5_set_attr(group.getId(), ATTR_TYPE, std::string(REPR_RANGE));
    hid_t a = save_hid(obj.attr("start"), subpath + "start");
    hid_t b = save_hid(obj.attr("stop"), subpath + "stop");
    hid_t c = save_hid(obj.attr("step"), subpath + "step");
    if (a >= 0)
        H5Idec_ref(a);
    if (b >= 0)
        H5Idec_ref(b);
    if (c >= 0)
        H5Idec_ref(c);
    hid_t id = group.getId();
    H5Iinc_ref(id);
    return id;
}

hid_t
Hdf5Saver::save_dtype(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    auto [group, subpath] = create_group_for_obj(path, obj);
    h5_set_attr(group.getId(), ATTR_TYPE, std::string(REPR_DTYPE));
    py::object name = py::getattr(obj, "name", py::str("void"));
    h5_set_attr(group.getId(), "name", name);
    hid_t descr = save_hid(obj.attr("descr"), subpath + "descr");
    if (descr >= 0)
        H5Idec_ref(descr);
    hid_t id = group.getId();
    H5Iinc_ref(id);
    return id;
}

hid_t
Hdf5Saver::save_ignored(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)obj;
    (void)path;
    (void)type_repr;
    return H5I_INVALID_HID;
}

hid_t
Hdf5Saver::save_global(py::object obj, std::string const& path, std::string const& type_repr)
{
    std::string module = obj.attr("__module__").cast<std::string>();
    std::string qualname = obj.attr("__qualname__").cast<std::string>();
    py::object obj2;
    try {
        obj2 = find_global(module, qualname);
    } catch (py::error_already_set const&) {
        throw Hdf5ExportError("Can't export object: it's not found as " + qualname +
                              " in module " + module);
    }
    if (!obj2.is(obj))
        throw Hdf5ExportError("Can't export object: it's not the same object as " + qualname +
                              " in module " + module);
    std::string full_name = qualname + " in " + module;
    h5_write_dataset(native_group, path, full_name);
    hid_t id = h5_open(native_group, path);
    h5_set_attr(id, ATTR_TYPE, type_repr);
    h5_set_attr(id, ATTR_CLASS, qualname);
    h5_set_attr(id, ATTR_MODULE, module);
    memorize_save(id, obj);
    return id;
}

} // namespace hdf5_io
