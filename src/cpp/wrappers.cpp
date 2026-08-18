#include <hdf5_io/hdf5_io.h>

#include <pybind11/pybind11.h>

namespace hdf5_io {

void
save(py::object data, std::string filename, std::string mode)
{
    if (filename.ends_with(".pkl")) {
        py::object f = py::module_::import("builtins").attr("open")(filename, mode + "b");
        py::object closer = f.attr("__enter__")();
        try {
            py::module_::import("pickle").attr("dump")(data, closer);
            f.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (...) {
            f.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        return;
    }
    if (filename.ends_with(".pklz")) {
        py::object f = py::module_::import("gzip").attr("open")(filename, mode + "b");
        py::object closer = f.attr("__enter__")();
        try {
            py::module_::import("pickle").attr("dump")(data, closer);
            f.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (...) {
            f.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        return;
    }
    if (filename.ends_with(".hdf5") || filename.ends_with(".h5")) {
        py::object File = py::module_::import("h5py").attr("File");
        py::object f = File(filename, mode);
        py::object closer = f.attr("__enter__")();
        try {
            save_to_hdf5(closer, data);
            f.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (...) {
            f.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        return;
    }
    throw std::invalid_argument("Don't recognise file ending of " + filename);
}

py::object
load(std::string filename)
{
    if (filename.ends_with(".pkl")) {
        py::object f = py::module_::import("builtins").attr("open")(filename, "rb");
        py::object closer = f.attr("__enter__")();
        py::object data;
        try {
            data = py::module_::import("pickle").attr("load")(closer);
            f.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (...) {
            f.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        return data;
    }
    if (filename.ends_with(".pklz")) {
        py::object f = py::module_::import("gzip").attr("open")(filename, "rb");
        py::object closer = f.attr("__enter__")();
        py::object data;
        try {
            data = py::module_::import("pickle").attr("load")(closer);
            f.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (...) {
            f.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        return data;
    }
    if (filename.ends_with(".hdf5") || filename.ends_with(".h5")) {
        py::object File = py::module_::import("h5py").attr("File");
        py::object f = File(filename, "r");
        py::object closer = f.attr("__enter__")();
        py::object data;
        try {
            data = load_from_hdf5(closer);
            f.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (...) {
            f.attr("__exit__")(py::none(), py::none(), py::none());
            throw;
        }
        return data;
    }
    throw std::invalid_argument("Don't recognise file ending of " + filename);
}

py::object
save_to_hdf5(py::object h5group, py::object obj, std::string path)
{
    return Hdf5Saver(std::move(h5group)).save(std::move(obj), std::move(path));
}

py::object
load_from_hdf5(py::object h5group, py::object path, bool ignore_unknown, py::object exclude)
{
    return Hdf5Loader(std::move(h5group), ignore_unknown, std::move(exclude))
      .load(std::move(path));
}

std::unordered_map<PyTypeObject*, std::pair<SaveMethod, char const*>>&
cpp_save_dispatch()
{
    static std::unordered_map<PyTypeObject*, std::pair<SaveMethod, char const*>> table;
    return table;
}

std::unordered_map<std::string, LoadMethod>&
cpp_load_dispatch()
{
    static std::unordered_map<std::string, LoadMethod> table;
    return table;
}

void
init_dispatch_tables(py::module_& m)
{
    py::object saver_cls = m.attr("Hdf5Saver");
    py::object loader_cls = m.attr("Hdf5Loader");
    py::dict dispatch_save;
    py::dict dispatch_load;
    py::module_ np = py::module_::import("numpy");
    py::module_ builtins = py::module_::import("builtins");
    py::module_ types = py::module_::import("types");

    auto add_save = [&](py::object ty, py::object method, char const* repr, SaveMethod cpp_fn) {
        dispatch_save[ty] = py::make_tuple(method, py::str(repr));
        cpp_save_dispatch()[reinterpret_cast<PyTypeObject*>(ty.ptr())] = { cpp_fn, repr };
    };
    auto add_load = [&](char const* repr, py::object method, py::object info, LoadMethod cpp_fn) {
        dispatch_load[py::str(repr)] = py::make_tuple(method, info);
        cpp_load_dispatch()[repr] = cpp_fn;
    };

    add_save(py::type::of(py::none()), saver_cls.attr("save_none"), REPR_NONE, &Hdf5Saver::save_none);

    py::list types_for_datasets;
    auto add_dataset = [&](py::object ty, char const* repr) {
        types_for_datasets.append(py::make_tuple(ty, py::str(repr)));
        add_save(ty, saver_cls.attr("save_dataset"), repr, &Hdf5Saver::save_dataset);
        add_load(repr, loader_cls.attr("load_dataset"), ty, &Hdf5Loader::load_dataset);
    };
    add_dataset(np.attr("ndarray"), REPR_ARRAY);
    add_dataset(builtins.attr("int"), REPR_INT);
    add_dataset(builtins.attr("float"), REPR_FLOAT);
    add_dataset(builtins.attr("str"), REPR_STR);
    add_dataset(builtins.attr("bytes"), REPR_BYTES);
    add_dataset(builtins.attr("complex"), REPR_COMPLEX);
    add_dataset(np.attr("int64"), REPR_INT64);
    add_dataset(np.attr("float64"), REPR_FLOAT64);
    add_dataset(np.attr("complex128"), REPR_COMPLEX128);
    add_dataset(np.attr("int32"), REPR_INT32);
    add_dataset(np.attr("float32"), REPR_FLOAT32);
    add_dataset(np.attr("complex64"), REPR_COMPLEX64);
    add_dataset(np.attr("bool_"), REPR_BOOL);
    add_dataset(builtins.attr("bool"), REPR_BOOL);

    m.attr("TYPES_FOR_HDF5_DATASETS") = py::tuple(types_for_datasets);

    add_save(np.attr("ma").attr("MaskedArray"),
             saver_cls.attr("save_masked_array"),
             REPR_MASKED_ARRAY,
             &Hdf5Saver::save_masked_array);
    add_save(builtins.attr("list"), saver_cls.attr("save_iterable"), REPR_LIST, &Hdf5Saver::save_iterable);
    add_save(builtins.attr("tuple"),
             saver_cls.attr("save_iterable"),
             REPR_TUPLE,
             &Hdf5Saver::save_iterable);
    add_save(builtins.attr("set"), saver_cls.attr("save_iterable"), REPR_SET, &Hdf5Saver::save_iterable);
    add_save(builtins.attr("dict"), saver_cls.attr("save_dict"), REPR_DICT_GENERAL, &Hdf5Saver::save_dict);
    add_save(builtins.attr("range"), saver_cls.attr("save_range"), REPR_RANGE, &Hdf5Saver::save_range);

    py::object dtype_type = np.attr("dtype");
    for (auto t : dtype_type.attr("__subclasses__")()) {
        py::object ty = py::reinterpret_borrow<py::object>(t);
        std::string name = ty.attr("__name__").cast<std::string>();
        if (!name.empty() && name.front() == '_') {
            for (auto t2 : ty.attr("__subclasses__")())
                add_save(py::reinterpret_borrow<py::object>(t2),
                         saver_cls.attr("save_dtype"),
                         REPR_DTYPE,
                         &Hdf5Saver::save_dtype);
        } else {
            add_save(ty, saver_cls.attr("save_dtype"), REPR_DTYPE, &Hdf5Saver::save_dtype);
        }
    }

    add_save(m.attr("Hdf5Ignored"), saver_cls.attr("save_ignored"), REPR_IGNORED, &Hdf5Saver::save_ignored);
    add_save(types.attr("FunctionType"),
             saver_cls.attr("save_global"),
             REPR_FUNCTION,
             &Hdf5Saver::save_global);
    add_save(types.attr("BuiltinFunctionType"),
             saver_cls.attr("save_global"),
             REPR_FUNCTION,
             &Hdf5Saver::save_global);
    add_save(builtins.attr("type"), saver_cls.attr("save_global"), REPR_CLASS, &Hdf5Saver::save_global);
    py::object pybind11_type = builtins.attr("type")(saver_cls);
    if (!pybind11_type.is(builtins.attr("type")))
        add_save(pybind11_type, saver_cls.attr("save_global"), REPR_CLASS, &Hdf5Saver::save_global);

    add_load(REPR_NONE, loader_cls.attr("load_none"), py::none(), &Hdf5Loader::load_none);
    add_load(REPR_STR, loader_cls.attr("load_str"), builtins.attr("str"), &Hdf5Loader::load_str);
    add_load(REPR_INT_AS_STR,
             loader_cls.attr("load_converted_to_str"),
             builtins.attr("int"),
             &Hdf5Loader::load_converted_to_str);
    add_load(REPR_MASKED_ARRAY,
             loader_cls.attr("load_masked_array"),
             py::str(REPR_MASKED_ARRAY),
             &Hdf5Loader::load_masked_array);
    add_load(REPR_LIST, loader_cls.attr("load_list"), py::str(REPR_LIST), &Hdf5Loader::load_list);
    add_load(REPR_SET, loader_cls.attr("load_set"), py::str(REPR_SET), &Hdf5Loader::load_set);
    add_load(REPR_TUPLE, loader_cls.attr("load_tuple"), py::str(REPR_TUPLE), &Hdf5Loader::load_tuple);
    add_load(REPR_DICT_GENERAL,
             loader_cls.attr("load_general_dict"),
             py::str(REPR_DICT_GENERAL),
             &Hdf5Loader::load_general_dict);
    add_load(REPR_DICT_SIMPLE,
             loader_cls.attr("load_simple_dict"),
             py::str(REPR_DICT_SIMPLE),
             &Hdf5Loader::load_simple_dict);
    add_load(REPR_RANGE, loader_cls.attr("load_range"), py::str(REPR_RANGE), &Hdf5Loader::load_range);
    add_load(REPR_DTYPE, loader_cls.attr("load_dtype"), py::str(REPR_DTYPE), &Hdf5Loader::load_dtype);
    add_load(REPR_HDF5EXPORTABLE,
             loader_cls.attr("load_hdf5exportable"),
             py::str(REPR_HDF5EXPORTABLE),
             &Hdf5Loader::load_hdf5exportable);
    add_load(REPR_IGNORED, loader_cls.attr("load_ignored"), py::str(REPR_IGNORED), &Hdf5Loader::load_ignored);
    add_load(REPR_FUNCTION, loader_cls.attr("load_global"), py::str(REPR_FUNCTION), &Hdf5Loader::load_global);
    add_load(REPR_CLASS, loader_cls.attr("load_global"), py::str(REPR_CLASS), &Hdf5Loader::load_global);
    add_load(REPR_GLOBAL, loader_cls.attr("load_global"), py::str(REPR_GLOBAL), &Hdf5Loader::load_global);
    add_load(REPR_REDUCE, loader_cls.attr("load_reduce"), py::str(REPR_REDUCE), &Hdf5Loader::load_reduce);

    saver_cls.attr("dispatch_save") = dispatch_save;
    loader_cls.attr("dispatch_load") = dispatch_load;
}

} // namespace hdf5_io
