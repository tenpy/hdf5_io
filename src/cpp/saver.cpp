#include <hdf5_io/h5_bridge.h>
#include <hdf5_io/hdf5_io.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <utility>

namespace hdf5_io {

namespace {

py::object
py_id(py::handle obj)
{
    return py::module_::import("builtins").attr("id")(obj);
}

py::object
self_obj(Hdf5Saver* self)
{
    return py::cast(self, py::return_value_policy::reference);
}

} // namespace

Hdf5Saver::Hdf5Saver(py::object h5group_, py::object format_selection_)
  : h5group(std::move(h5group_))
  , memo_save(py::dict())
{
    if (format_selection_.is_none())
        format_selection = py::dict();
    else
        format_selection = py::reinterpret_borrow<py::dict>(format_selection_);
}

py::dict
Hdf5Saver::dispatch_save() const
{
    return py::type::of<Hdf5Saver>().attr("dispatch_save");
}

py::object
Hdf5Saver::save(py::object obj, std::string path)
{
    py::object obj_id = py_id(obj);
    py::object in_memo = memo_save.attr("get")(obj_id);
    if (!in_memo.is_none()) {
        py::object h5gr = in_memo[py::int_(0)];
        h5_set_item(h5group, path, h5gr);
        return h5gr;
    }

    py::object disp = dispatch_save().attr("get")(py::type::of(obj));
    if (!disp.is_none()) {
        py::object f = disp[py::int_(0)];
        py::object type_repr = disp[py::int_(1)];
        return f(self_obj(this), obj, path, type_repr);
    }

    py::object obj_save_hdf5 = py::getattr(obj, "save_hdf5", py::none());
    if (!obj_save_hdf5.is_none()) {
        auto [h5gr, subpath] = create_group_for_obj(path, obj);
        h5_set_attr(h5gr, ATTR_TYPE, py::str(REPR_HDF5EXPORTABLE));
        h5_set_attr(h5gr, ATTR_CLASS, obj.attr("__class__").attr("__qualname__"));
        h5_set_attr(h5gr, ATTR_MODULE, obj.attr("__class__").attr("__module__"));
        obj_save_hdf5(self_obj(this), h5gr, subpath);
        return h5gr;
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
        return save_reduce(
          func, args, state, listitems, dictitems, state_setter, obj, py::str(path));
    }

    throw Hdf5ExportError("Don't know how to save object of type " +
                          py::repr(py::type::of(obj)).cast<std::string>());
}

std::pair<py::object, std::string>
Hdf5Saver::create_group_for_obj(std::string const& path, py::object obj)
{
    py::object gr;
    if (path == "/")
        gr = h5group[py::str(path)];
    else
        gr = h5_create_group(h5group, path);
    std::string subpath = (!path.empty() && path.back() == '/') ? path : (path + "/");
    memorize_save(gr, obj);
    return { gr, subpath };
}

void
Hdf5Saver::memorize_save(py::object h5gr, py::object obj)
{
    py::object obj_id = py_id(obj);
    if (memo_save.contains(obj_id))
        throw Hdf5ExportError("object already memorized");
    memo_save[obj_id] = py::make_tuple(h5gr, obj);
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
    auto [h5gr, subpath] = create_group_for_obj(p, obj);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(REPR_REDUCE));
    save(func, subpath + "func");
    save(args, subpath + "args");
    if (!state.is_none())
        save(state, subpath + "state");
    if (!listitems.is_none())
        save(state, subpath + "listitems");
    if (!dictitems.is_none())
        save(state, subpath + "dictitems");
    if (!state_setter.is_none())
        save(state, subpath + "state_setter");
    return h5gr;
}

py::object
Hdf5Saver::save_none(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    h5_write_dataset(h5group, path, py::str(REPR_NONE));
    py::object h5gr = h5py_getitem(h5group, path);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(REPR_NONE));
    memorize_save(h5gr, obj);
    return h5gr;
}

py::object
Hdf5Saver::save_dataset(py::object obj, std::string const& path, std::string type_repr)
{
    try {
        h5_write_dataset(h5group, path, obj);
    } catch (py::error_already_set& e) {
        if (type_repr != REPR_INT || !e.matches(PyExc_TypeError))
            throw;
        std::string msg = e.what();
        if (msg.find("no native HDF5 equivalent") == std::string::npos)
            throw;
        obj = py::str(obj);
        type_repr = REPR_INT_AS_STR;
        h5_write_dataset(h5group, path, obj);
    } catch (Hdf5ExportError const& e) {
        if (type_repr != REPR_INT ||
            std::string(e.what()).find("no native HDF5 equivalent") == std::string::npos)
            throw;
        obj = py::str(obj);
        type_repr = REPR_INT_AS_STR;
        h5_write_dataset(h5group, path, obj);
    } catch (py::type_error const& e) {
        if (type_repr != REPR_INT ||
            std::string(e.what()).find("no native HDF5 equivalent") == std::string::npos)
            throw;
        obj = py::str(obj);
        type_repr = REPR_INT_AS_STR;
        h5_write_dataset(h5group, path, obj);
    }
    py::object h5gr = h5py_getitem(h5group, path);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(type_repr));
    memorize_save(h5gr, obj);
    return h5gr;
}

py::object
Hdf5Saver::save_masked_array(py::object obj, std::string const& path, std::string const& type_repr)
{
    py::object filled = obj.attr("filled")();
    py::object fill_value = obj.attr("fill_value");
    py::object np = py::module_::import("numpy");
    py::object h5gr;
    py::object mask = obj.attr("mask");
    py::object cmp = filled.attr("__eq__")(fill_value);
    py::object mask_cmp = cmp.attr("__eq__")(mask);
    if (np.attr("any")(mask_cmp).cast<bool>()) {
        auto created = create_group_for_obj(path, obj);
        h5gr = created.first;
        std::string subpath = created.second;
        h5_write_dataset(h5gr, "data", obj.attr("data"));
        h5_write_dataset(h5gr, "mask", obj.attr("mask"));
        h5_set_attr(h5gr, "saved_mask", py::bool_(true));
    } else {
        h5_write_dataset(h5group, path, filled);
        h5gr = h5py_getitem(h5group, path);
        h5_set_attr(h5gr, "saved_mask", py::bool_(false));
        memorize_save(h5gr, obj);
    }
    h5_set_attr(h5gr, ATTR_TYPE, py::str(type_repr));
    h5_set_attr(h5gr, "fill_value", fill_value);
    return h5gr;
}

py::object
Hdf5Saver::save_iterable(py::object obj, std::string const& path, std::string const& type_repr)
{
    auto [h5gr, subpath] = create_group_for_obj(path, obj);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(type_repr));
    save_iterable_content(obj, h5gr, subpath);
    return h5gr;
}

void
Hdf5Saver::save_iterable_content(py::object obj, py::object h5gr, std::string const& subpath)
{
    py::ssize_t n = py::len(obj);
    h5_set_attr(h5gr, ATTR_LEN, py::int_(n));
    py::ssize_t i = 0;
    for (auto elem : obj) {
        save(py::reinterpret_borrow<py::object>(elem), subpath + std::to_string(i));
        ++i;
    }
}

py::object
Hdf5Saver::save_dict(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    auto [h5gr, subpath] = create_group_for_obj(path, obj);
    std::string tr = save_dict_content(obj, h5gr, subpath);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(tr));
    return h5gr;
}

std::string
Hdf5Saver::save_dict_content(py::object obj, py::object h5gr, std::string const& subpath)
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
            save(kv[1], subpath + kv[0].cast<std::string>());
        }
        return REPR_DICT_SIMPLE;
    }
    save_iterable(obj.attr("keys")(), subpath + "keys", REPR_LIST);
    save_iterable(obj.attr("values")(), subpath + "values", REPR_LIST);
    return REPR_DICT_GENERAL;
}

py::object
Hdf5Saver::save_range(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    auto [h5gr, subpath] = create_group_for_obj(path, obj);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(REPR_RANGE));
    save(obj.attr("start"), subpath + "start");
    save(obj.attr("stop"), subpath + "stop");
    save(obj.attr("step"), subpath + "step");
    return h5gr;
}

py::object
Hdf5Saver::save_dtype(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)type_repr;
    auto [h5gr, subpath] = create_group_for_obj(path, obj);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(REPR_DTYPE));
    py::object name = py::getattr(obj, "name", py::str("void"));
    h5_set_attr(h5gr, "name", name);
    save(obj.attr("descr"), subpath + "descr");
    return h5gr;
}

py::object
Hdf5Saver::save_ignored(py::object obj, std::string const& path, std::string const& type_repr)
{
    (void)obj;
    (void)path;
    (void)type_repr;
    return py::none();
}

py::object
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
    h5_write_dataset(h5group, path, py::str(full_name));
    py::object h5gr = h5py_getitem(h5group, path);
    h5_set_attr(h5gr, ATTR_TYPE, py::str(type_repr));
    h5_set_attr(h5gr, ATTR_CLASS, py::str(qualname));
    h5_set_attr(h5gr, ATTR_MODULE, py::str(module));
    memorize_save(h5gr, obj);
    return h5gr;
}

} // namespace hdf5_io
