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
ensure_subpath(std::string const& path)
{
    return (!path.empty() && path.back() == '/') ? path : (path + "/");
}

py::object
convert_scalar(py::object raw, std::string const& type_repr)
{
    py::module_ np = py::module_::import("numpy");
    py::module_ builtins = py::module_::import("builtins");
    if (type_repr == REPR_INT)
        return builtins.attr("int")(raw);
    if (type_repr == REPR_FLOAT)
        return builtins.attr("float")(raw);
    if (type_repr == REPR_COMPLEX)
        return builtins.attr("complex")(raw);
    if (type_repr == REPR_BOOL)
        return builtins.attr("bool")(raw);
    if (type_repr == REPR_BYTES)
        return builtins.attr("bytes")(raw);
    if (type_repr == REPR_INT64)
        return np.attr("int64")(raw);
    if (type_repr == REPR_INT32)
        return np.attr("int32")(raw);
    if (type_repr == REPR_FLOAT64)
        return np.attr("float64")(raw);
    if (type_repr == REPR_FLOAT32)
        return np.attr("float32")(raw);
    if (type_repr == REPR_COMPLEX128)
        return np.attr("complex128")(raw);
    if (type_repr == REPR_COMPLEX64)
        return np.attr("complex64")(raw);
    return raw;
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
        std::string p = path.cast<std::string>();
        if (!h5_contains(native_group, p)) {
            py::module_::import("warnings")
              .attr("warn")(py::str("can't exclude {0!r} from loading: not existent in h5group")
                              .attr("format")(path));
            continue;
        }
        hid_t obj = h5_open(native_group, p);
        memorize_load(obj, py::cast(Hdf5Ignored(p)));
        H5Idec_ref(obj);
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
    hid_t obj;
    std::string path_str;
    if (path.is_none()) {
        obj = native_group.getId();
        H5Iinc_ref(obj);
        path_str = h5_object_name(obj);
    } else {
        path_str = path.cast<std::string>();
        obj = h5_open(native_group, path_str);
    }
    std::string subpath = ensure_subpath(path_str);
    py::object result = load_from(obj, subpath);
    H5Idec_ref(obj);
    return result;
}

py::object
Hdf5Loader::load_from(hid_t h5obj, std::string subpath)
{
    std::string key = object_token(h5obj);
    auto in_memo = memo_load_objects.find(key);
    if (in_memo != memo_load_objects.end())
        return in_memo->second;

    py::object type_repr_obj = get_attr(h5obj, ATTR_TYPE);
    std::string type_repr = py::str(type_repr_obj).cast<std::string>();

    auto const& cpp_disp = cpp_load_dispatch();
    auto it = cpp_disp.find(type_repr);
    if (it != cpp_disp.end())
        return (this->*(it->second))(h5obj, type_repr, subpath);

    py::object disp = dispatch_load().attr("get")(type_repr_obj);
    if (disp.is_none()) {
        throw Hdf5ImportError("Unknown type " + type_repr + " while loading hdf5 dataset " +
                              h5_object_name(h5obj));
    }
    py::object f = disp[py::int_(0)];
    py::object type_info = disp[py::int_(1)];
    py::object h5gr = h5py_from_hid(h5obj);
    return f(self_obj(this), h5gr, type_info, subpath);
}

void
Hdf5Loader::memorize_load(hid_t h5obj, py::object obj)
{
    memo_load_objects.try_emplace(object_token(h5obj), std::move(obj));
}

void
Hdf5Loader::memorize_load(py::object h5gr, py::object obj)
{
    memorize_load(hid_from_h5py(h5gr), std::move(obj));
}

py::object
Hdf5Loader::get_all_hdf5_keys(py::object h5_group)
{
    hid_t loc;
    bool close = false;
    if (h5_group.is_none()) {
        loc = native_group.getId();
    } else {
        loc = hid_from_h5py(h5_group);
    }
    auto names = h5_link_names(loc);
    py::dict results;
    bool any_group = false;
    for (auto const& name : names) {
        hid_t child = h5_open(loc, name);
        if (H5Iget_type(child) == H5I_GROUP) {
            results[py::str(name)] = get_all_hdf5_keys(h5py_from_hid(child));
            any_group = true;
        } else {
            results[py::str(name)] = h5py_from_hid(child);
        }
        H5Idec_ref(child);
        (void)close;
    }
    if (!any_group)
        return py::set(results.attr("keys")());
    return results;
}

py::object
Hdf5Loader::get_attr(hid_t h5obj, std::string const& attr_name)
{
    py::object res = h5_get_attr(h5obj, attr_name);
    if (res.is_none())
        throw Hdf5ImportError("missing attribute " + attr_name + " for dataset " +
                              h5_object_name(h5obj));
    return res;
}

py::object
Hdf5Loader::get_attr(py::object h5gr, std::string const& attr_name)
{
    return get_attr(hid_from_h5py(h5gr), attr_name);
}

py::object
Hdf5Loader::load_none(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    py::object obj = py::none();
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_dataset(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)subpath;
    std::string repr = type_info;
    if (repr != REPR_ARRAY && repr != REPR_INT && repr != REPR_FLOAT && repr != REPR_COMPLEX &&
        repr != REPR_BOOL && repr != REPR_BYTES && repr != REPR_INT64 && repr != REPR_INT32 &&
        repr != REPR_FLOAT64 && repr != REPR_FLOAT32 && repr != REPR_COMPLEX128 &&
        repr != REPR_COMPLEX64)
        repr = py::str(get_attr(h5obj, ATTR_TYPE)).cast<std::string>();
    py::object obj;
    if (repr == REPR_BYTES)
        obj = py::bytes(h5_read_vlen_string(h5obj));
    else if (repr == REPR_ARRAY)
        obj = h5_read_array(h5obj);
    else {
        py::array arr = h5_read_array(h5obj);
        obj = convert_scalar(arr.attr("item")(), repr);
    }
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_str(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    py::object obj = py::str(h5_read_vlen_string(h5obj));
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_converted_to_str(hid_t h5obj,
                                  std::string const& type_info,
                                  std::string const& subpath)
{
    (void)subpath;
    py::object obj = py::str(h5_read_vlen_string(h5obj));
    obj = convert_scalar(obj, type_info == REPR_INT_AS_STR ? std::string(REPR_INT) : type_info);
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_masked_array(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    py::object np = py::module_::import("numpy");
    py::object fill_value = get_attr(h5obj, "fill_value");
    py::object saved_mask = get_attr(h5obj, "saved_mask");
    py::object obj;
    if (saved_mask.cast<bool>()) {
        hid_t data_id = h5_open(h5obj, "data");
        hid_t mask_id = h5_open(h5obj, "mask");
        py::object data = h5_read_array(data_id);
        py::object mask = h5_read_array(mask_id);
        H5Idec_ref(data_id);
        H5Idec_ref(mask_id);
        py::dict kw;
        kw["mask"] = mask;
        kw["fill_value"] = fill_value;
        obj = np.attr("ma").attr("MaskedArray")(data, **kw);
    } else {
        py::object filled = h5_read_array(h5obj);
        py::dict kw;
        kw["copy"] = false;
        obj = np.attr("ma").attr("masked_equal")(filled, fill_value, **kw);
    }
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_list(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::list obj;
    memorize_load(h5obj, obj);
    py::ssize_t length = get_attr(h5obj, ATTR_LEN).cast<py::ssize_t>();
    for (py::ssize_t i = 0; i < length; ++i)
        obj.append(load(py::str(subpath + std::to_string(i))));
    return obj;
}

py::object
Hdf5Loader::load_set(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::object obj = py::set();
    memorize_load(h5obj, obj);
    py::ssize_t length = get_attr(h5obj, ATTR_LEN).cast<py::ssize_t>();
    for (py::ssize_t i = 0; i < length; ++i)
        obj.attr("add")(load(py::str(subpath + std::to_string(i))));
    return obj;
}

py::object
Hdf5Loader::load_tuple(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::list obj;
    memorize_load(h5obj, obj);
    py::ssize_t length = get_attr(h5obj, ATTR_LEN).cast<py::ssize_t>();
    for (py::ssize_t i = 0; i < length; ++i)
        obj.append(load(py::str(subpath + std::to_string(i))));
    py::tuple tup(obj);
    memo_load_objects[object_token(h5obj)] = tup;
    return tup;
}

py::object
Hdf5Loader::load_dict(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    if (type_info == REPR_DICT_GENERAL)
        return load_general_dict(h5obj, type_info, subpath);
    if (type_info == REPR_DICT_SIMPLE)
        return load_simple_dict(h5obj, type_info, subpath);
    throw std::invalid_argument("can't interpret type_info " + type_info);
}

py::object
Hdf5Loader::load_general_dict(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::dict obj;
    memorize_load(h5obj, obj);
    hid_t keys_id = h5_open(h5obj, "keys");
    hid_t values_id = h5_open(h5obj, "values");
    py::object keys = load_list(keys_id, REPR_LIST, subpath + "keys/");
    py::object values = load_list(values_id, REPR_LIST, subpath + "values/");
    H5Idec_ref(keys_id);
    H5Idec_ref(values_id);
    obj.attr("update")(py::module_::import("builtins").attr("zip")(keys, values));
    return obj;
}

py::object
Hdf5Loader::load_simple_dict(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::dict obj;
    memorize_load(h5obj, obj);
    for (auto const& key : h5_link_names(h5obj))
        obj[py::str(key)] = load(py::str(subpath + key));
    return obj;
}

py::object
Hdf5Loader::load_range(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::object start = load(py::str(subpath + "start"));
    py::object stop = load(py::str(subpath + "stop"));
    py::object step = load(py::str(subpath + "step"));
    py::object obj = py::module_::import("builtins").attr("range")(start, stop, step);
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_dtype(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::object np = py::module_::import("numpy");
    std::string name = py::str(get_attr(h5obj, "name")).cast<std::string>();
    py::object obj;
    if (name.rfind("void", 0) == 0) {
        py::object descr = load(py::str(subpath + "descr"));
        obj = np.attr("dtype")(descr);
    } else {
        obj = np.attr("dtype")(name);
    }
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_hdf5exportable(hid_t h5obj,
                                std::string const& type_info,
                                std::string const& subpath)
{
    (void)type_info;
    std::string module_name = py::str(get_attr(h5obj, ATTR_MODULE)).cast<std::string>();
    std::string class_name = py::str(get_attr(h5obj, ATTR_CLASS)).cast<std::string>();
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
    py::object h5gr = h5py_from_hid(h5obj);
    return cls.attr("from_hdf5")(self_obj(this), h5gr, subpath);
}

py::object
Hdf5Loader::load_ignored(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    return py::cast(Hdf5Ignored(h5_object_name(h5obj)));
}

py::object
Hdf5Loader::load_global(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    (void)subpath;
    std::string module_name = py::str(get_attr(h5obj, ATTR_MODULE)).cast<std::string>();
    std::string class_name = py::str(get_attr(h5obj, ATTR_CLASS)).cast<std::string>();
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
    memorize_load(h5obj, obj);
    return obj;
}

py::object
Hdf5Loader::load_reduce(hid_t h5obj, std::string const& type_info, std::string const& subpath)
{
    (void)type_info;
    py::object func = load(py::str(subpath + "func"));
    py::object args = load(py::str(subpath + "args"));
    py::object obj = func(*py::reinterpret_borrow<py::tuple>(args));
    memorize_load(h5obj, obj);
    if (h5_contains(h5obj, "state")) {
        py::object state = load(py::str(subpath + "state"));
        if (h5_contains(h5obj, "state_setter")) {
            py::object state_setter = load(py::str(subpath + "state_setter"));
            obj = state_setter(obj, state);
            memorize_load(h5obj, obj);
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
    if (h5_contains(h5obj, "listitems")) {
        py::object listitems = load(py::str(subpath + "listitems"));
        for (auto item : listitems)
            obj.attr("append")(item);
    }
    if (h5_contains(h5obj, "dictitems")) {
        py::object dictitems = load(py::str(subpath + "dictitems"));
        for (auto item : dictitems) {
            py::tuple kv = py::reinterpret_borrow<py::tuple>(item);
            obj[kv[0]] = kv[1];
        }
    }
    return obj;
}

} // namespace hdf5_io
