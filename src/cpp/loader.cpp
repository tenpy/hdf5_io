#include <hdf5_io/h5_bridge.h>
#include <hdf5_io/hdf5_io.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace hdf5_io {

namespace {

py::object
self_obj(Hdf5Loader* self)
{
    return py::cast(self, py::return_value_policy::reference);
}

std::string
h5_id(py::handle h5gr)
{
    return object_token(h5gr);
}

} // namespace

Hdf5Loader::Hdf5Loader(py::object h5group_, bool ignore_unknown_, py::object exclude)
  : h5group(std::move(h5group_))
  , ignore_unknown(ignore_unknown_)
  , native_group(wrap_group(h5group))
{
    if (exclude.is_none())
        return;
    for (auto path : exclude) {
        try {
            py::object data = h5group[path];
            memorize_load(data, py::cast(Hdf5Ignored(path.cast<std::string>())));
        } catch (py::error_already_set& e) {
            if (!e.matches(PyExc_KeyError))
                throw;
            e.restore();
            PyErr_Clear();
            py::module_::import("warnings")
              .attr("warn")(py::str("can't exclude {0!r} from loading: not existent in h5group")
                              .attr("format")(path));
        }
    }
}

py::dict
Hdf5Loader::memo_load() const
{
    py::dict out;
    for (auto const& [token, obj] : memo_load_objects)
        out[py::str(token)] = obj;
    return out;
}

py::dict
Hdf5Loader::dispatch_load() const
{
    return py::type::of<Hdf5Loader>().attr("dispatch_load");
}

py::object
Hdf5Loader::load(py::object path)
{
    py::object h5gr;
    std::string path_str;
    if (path.is_none()) {
        h5gr = h5group;
        path_str = h5group.attr("name").cast<std::string>();
    } else {
        path_str = path.cast<std::string>();
        h5gr = h5group[py::str(path_str)];
    }
    std::string subpath =
      (!path_str.empty() && path_str.back() == '/') ? path_str : (path_str + "/");
    auto key = h5_id(h5gr);
    auto in_memo = memo_load_objects.find(key);
    if (in_memo != memo_load_objects.end())
        return in_memo->second;

    py::object type_repr = get_attr(h5gr, ATTR_TYPE);
    py::object disp = dispatch_load().attr("get")(type_repr);
    if (disp.is_none()) {
        throw Hdf5ImportError("Unknown type " + py::repr(type_repr).cast<std::string>() +
                              " while loading hdf5 dataset " +
                              h5gr.attr("name").cast<std::string>());
    }
    py::object f = disp[py::int_(0)];
    py::object type_info = disp[py::int_(1)];
    return f(self_obj(this), h5gr, type_info, subpath);
}

void
Hdf5Loader::memorize_load(py::object h5gr, py::object obj)
{
    memo_load_objects.try_emplace(h5_id(h5gr), std::move(obj));
}

py::object
Hdf5Loader::get_all_hdf5_keys(py::object h5_group)
{
    if (h5_group.is_none())
        h5_group = h5group;
    auto h5py = py::module_::import("h5py");
    py::dict results;
    py::bool_ any_group(false);
    for (auto key : h5_group.attr("keys")()) {
        py::object child = h5_group[key];
        if (py::isinstance(child, h5py.attr("Group"))) {
            results[key] = get_all_hdf5_keys(child);
            any_group = py::bool_(true);
        } else {
            results[key] = child;
        }
    }
    if (!any_group.cast<bool>())
        return py::set(results.attr("keys")());
    return results;
}

py::object
Hdf5Loader::get_attr(py::object h5gr, std::string const& attr_name)
{
    py::object res = h5_get_attr(h5gr, attr_name);
    if (res.is_none())
        throw Hdf5ImportError("missing attribute " + attr_name + " for dataset " +
                              h5gr.attr("name").cast<std::string>());
    return res;
}

py::object
Hdf5Loader::load_none(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    py::object obj = py::none();
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_dataset(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)subpath;
    py::object np = py::module_::import("numpy");
    py::object obj;
    if (type_info.is(np.attr("ndarray")))
        obj = h5gr[py::ellipsis()];
    else {
        obj = h5gr[py::make_tuple()];
        obj = type_info(obj);
    }
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_str(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    py::object obj = h5gr.attr("asstr")()[py::make_tuple()];
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_converted_to_str(py::object h5gr,
                                  py::object type_info,
                                  std::string const& subpath)
{
    (void)subpath;
    py::object obj = h5gr.attr("asstr")()[py::make_tuple()];
    obj = type_info(obj);
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_masked_array(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    py::object np = py::module_::import("numpy");
    py::object fill_value = get_attr(h5gr, "fill_value");
    py::object saved_mask = get_attr(h5gr, "saved_mask");
    py::object obj;
    if (saved_mask.cast<bool>()) {
        py::object data = h5gr[py::str("data")][py::make_tuple()];
        py::object mask = h5gr[py::str("mask")][py::make_tuple()];
        py::dict kw;
        kw["mask"] = mask;
        kw["fill_value"] = fill_value;
        obj = np.attr("ma").attr("MaskedArray")(data, **kw);
    } else {
        py::object filled = h5gr[py::make_tuple()];
        py::dict kw;
        kw["copy"] = false;
        obj = np.attr("ma").attr("masked_equal")(filled, fill_value, **kw);
    }
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_list(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    py::list obj;
    memorize_load(h5gr, obj);
    py::ssize_t length = get_attr(h5gr, ATTR_LEN).cast<py::ssize_t>();
    for (py::ssize_t i = 0; i < length; ++i)
        obj.append(load(py::str(subpath + std::to_string(i))));
    return obj;
}

py::object
Hdf5Loader::load_set(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    py::object obj = py::set();
    memorize_load(h5gr, obj);
    py::ssize_t length = get_attr(h5gr, ATTR_LEN).cast<py::ssize_t>();
    for (py::ssize_t i = 0; i < length; ++i)
        obj.attr("add")(load(py::str(subpath + std::to_string(i))));
    return obj;
}

py::object
Hdf5Loader::load_tuple(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    py::list obj;
    memorize_load(h5gr, obj);
    py::ssize_t length = get_attr(h5gr, ATTR_LEN).cast<py::ssize_t>();
    for (py::ssize_t i = 0; i < length; ++i)
        obj.append(load(py::str(subpath + std::to_string(i))));
    py::tuple tup(obj);
    memo_load_objects[h5_id(h5gr)] = tup;
    return tup;
}

py::object
Hdf5Loader::load_dict(py::object h5gr, py::object type_info, std::string const& subpath)
{
    if (py::str(type_info).cast<std::string>() == REPR_DICT_GENERAL)
        return load_general_dict(h5gr, type_info, subpath);
    if (py::str(type_info).cast<std::string>() == REPR_DICT_SIMPLE)
        return load_simple_dict(h5gr, type_info, subpath);
    throw std::invalid_argument("can't interpret type_info " +
                                py::repr(type_info).cast<std::string>());
}

py::object
Hdf5Loader::load_general_dict(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    py::dict obj;
    memorize_load(h5gr, obj);
    py::object keys = load_list(h5gr[py::str("keys")], py::str(REPR_LIST), subpath + "keys/");
    py::object values =
      load_list(h5gr[py::str("values")], py::str(REPR_LIST), subpath + "values/");
    obj.attr("update")(py::module_::import("builtins").attr("zip")(keys, values));
    return obj;
}

py::object
Hdf5Loader::load_simple_dict(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    py::dict obj;
    memorize_load(h5gr, obj);
    for (auto k : h5gr.attr("keys")()) {
        std::string key = py::str(k).cast<std::string>();
        obj[k] = load(py::str(subpath + key));
    }
    return obj;
}

py::object
Hdf5Loader::load_range(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)h5gr;
    (void)type_info;
    py::object start = load(py::str(subpath + "start"));
    py::object stop = load(py::str(subpath + "stop"));
    py::object step = load(py::str(subpath + "step"));
    py::object obj = py::module_::import("builtins").attr("range")(start, stop, step);
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_dtype(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    py::object np = py::module_::import("numpy");
    std::string name = get_attr(h5gr, "name").cast<std::string>();
    py::object obj;
    if (name.rfind("void", 0) == 0) {
        py::object descr = load(py::str(subpath + "descr"));
        obj = np.attr("dtype")(descr);
    } else {
        obj = np.attr("dtype")(name);
    }
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_hdf5exportable(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    std::string module_name = get_attr(h5gr, ATTR_MODULE).cast<std::string>();
    std::string class_name = get_attr(h5gr, ATTR_CLASS).cast<std::string>();
    py::object cls;
    try {
        cls = find_global(module_name, class_name);
    } catch (py::error_already_set& e) {
        std::string msg = "Can't import class " + class_name + " from " + module_name;
        if (ignore_unknown) {
            py::module_::import("warnings").attr("warn")(msg);
            return py::cast(Hdf5Ignored(msg));
        }
        throw;
    }
    return Hdf5Exportable::from_hdf5(cls.cast<py::type>(), self_obj(this), h5gr, subpath);
}

py::object
Hdf5Loader::load_ignored(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    return py::cast(Hdf5Ignored(h5gr.attr("name").cast<std::string>()));
}

py::object
Hdf5Loader::load_global(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    std::string module_name = get_attr(h5gr, ATTR_MODULE).cast<std::string>();
    std::string class_name = get_attr(h5gr, ATTR_CLASS).cast<std::string>();
    py::object obj;
    try {
        obj = find_global(module_name, class_name);
    } catch (py::error_already_set& e) {
        std::string msg = "Can't import global " + class_name + " from " + module_name;
        if (ignore_unknown) {
            py::module_::import("warnings").attr("warn")(msg);
            return py::cast(Hdf5Ignored(msg));
        }
        throw;
    }
    memorize_load(h5gr, obj);
    return obj;
}

py::object
Hdf5Loader::load_reduce(py::object h5gr, py::object type_info, std::string const& subpath)
{
    (void)type_info;
    py::object func = load(py::str(subpath + "func"));
    py::object args = load(py::str(subpath + "args"));
    py::object obj = func(*py::reinterpret_borrow<py::tuple>(args));
    memorize_load(h5gr, obj);
    if (h5_contains(h5gr, "state")) {
        py::object state = load(py::str(subpath + "state"));
        if (h5_contains(h5gr, "state_setter")) {
            py::object state_setter = load(py::str(subpath + "state_setter"));
            obj = state_setter(obj, state);
            memorize_load(h5gr, obj);
        } else {
            py::object setstate = py::getattr(obj, "__setstate__", py::none());
            if (!setstate.is_none()) {
                setstate(state);
            } else {
                py::object slotstate = py::none();
                if (py::isinstance<py::tuple>(state) && py::len(state) == 2) {
                    py::tuple st = py::reinterpret_borrow<py::tuple>(state);
                    state = st[0];
                    slotstate = st[1];
                }
                if (!state.is_none() && py::bool_(state).cast<bool>()) {
                    py::object obj_dict = obj.attr("__dict__");
                    py::object intern = py::module_::import("sys").attr("intern");
                    for (auto item : state.attr("items")()) {
                        py::tuple kv = py::reinterpret_borrow<py::tuple>(item);
                        py::object k = kv[0];
                        if (py::type::of(k).is(py::type::of(py::str())))
                            obj_dict[intern(k)] = kv[1];
                        else
                            obj_dict[k] = kv[1];
                    }
                }
                if (!slotstate.is_none() && py::bool_(slotstate).cast<bool>()) {
                    for (auto item : slotstate.attr("items")()) {
                        py::tuple kv = py::reinterpret_borrow<py::tuple>(item);
                        py::setattr(obj, kv[0], kv[1]);
                    }
                }
            }
        }
    }
    if (h5_contains(h5gr, "listitems")) {
        py::object listitems = load(py::str(subpath + "listitems"));
        for (auto item : listitems)
            obj.attr("append")(item);
    }
    if (h5_contains(h5gr, "dictitems")) {
        py::object dictitems = load(py::str(subpath + "dictitems"));
        for (auto item : dictitems) {
            py::tuple kv = py::reinterpret_borrow<py::tuple>(item);
            obj[kv[0]] = kv[1];
        }
    }
    return obj;
}

} // namespace hdf5_io
