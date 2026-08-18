#include "py_hdf5_io.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace hdf5_io {

class PyHdf5Exportable
  : public Hdf5Exportable
  , public py::trampoline_self_life_support
{
  public:
    using Hdf5Exportable::Hdf5Exportable;

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string subpath) override
    {
        PYBIND11_OVERRIDE(void, Hdf5Exportable, save_hdf5, hdf5_saver, h5gr, subpath);
    }
};

void
bind_hdf5_io(py::module_& m)
{
    auto& format_err =
      py::register_exception<Hdf5FormatError>(m, "Hdf5FormatError", PyExc_Exception);
    format_err.doc() = "Common base class for errors regarding our HDF5 format.";
    auto& export_err =
      py::register_exception<Hdf5ExportError>(m, "Hdf5ExportError", format_err);
    export_err.doc() = "This exception is raised when something went wrong during export to hdf5.";
    auto& import_err =
      py::register_exception<Hdf5ImportError>(m, "Hdf5ImportError", format_err);
    import_err.doc() = "This exception is raised when something went wrong during import from hdf5.";

    m.attr("REPR_IGNORED") = REPR_IGNORED;
    m.attr("REPR_HDF5EXPORTABLE") = REPR_HDF5EXPORTABLE;
    m.attr("REPR_REDUCE") = REPR_REDUCE;
    m.attr("REPR_ARRAY") = REPR_ARRAY;
    m.attr("REPR_MASKED_ARRAY") = REPR_MASKED_ARRAY;
    m.attr("REPR_INT") = REPR_INT;
    m.attr("REPR_INT_AS_STR") = REPR_INT_AS_STR;
    m.attr("REPR_FLOAT") = REPR_FLOAT;
    m.attr("REPR_STR") = REPR_STR;
    m.attr("REPR_BYTES") = REPR_BYTES;
    m.attr("REPR_COMPLEX") = REPR_COMPLEX;
    m.attr("REPR_INT64") = REPR_INT64;
    m.attr("REPR_FLOAT64") = REPR_FLOAT64;
    m.attr("REPR_COMPLEX128") = REPR_COMPLEX128;
    m.attr("REPR_INT32") = REPR_INT32;
    m.attr("REPR_FLOAT32") = REPR_FLOAT32;
    m.attr("REPR_COMPLEX64") = REPR_COMPLEX64;
    m.attr("REPR_BOOL") = REPR_BOOL;
    m.attr("REPR_NONE") = REPR_NONE;
    m.attr("REPR_RANGE") = REPR_RANGE;
    m.attr("REPR_LIST") = REPR_LIST;
    m.attr("REPR_TUPLE") = REPR_TUPLE;
    m.attr("REPR_SET") = REPR_SET;
    m.attr("REPR_DICT_GENERAL") = REPR_DICT_GENERAL;
    m.attr("REPR_DICT_SIMPLE") = REPR_DICT_SIMPLE;
    m.attr("REPR_DTYPE") = REPR_DTYPE;
    m.attr("REPR_FUNCTION") = REPR_FUNCTION;
    m.attr("REPR_CLASS") = REPR_CLASS;
    m.attr("REPR_GLOBAL") = REPR_GLOBAL;
    m.attr("ATTR_TYPE") = ATTR_TYPE;
    m.attr("ATTR_CLASS") = ATTR_CLASS;
    m.attr("ATTR_MODULE") = ATTR_MODULE;
    m.attr("ATTR_LEN") = ATTR_LEN;
    m.attr("ATTR_FORMAT") = ATTR_FORMAT;

    m.def("valid_hdf5_path_component",
          &valid_hdf5_path_component,
          py::arg("name"),
          R"pydoc(
          Determine if `name` is a valid HDF5 path component.

          Conditions: String, no ``'/'``, and overall ``name != '.'``.
          )pydoc");
    m.def("find_global",
          &find_global,
          py::arg("module"),
          py::arg("qualified_name"),
          R"pydoc(
          Get the object of the `qualified_name` in a given python `module`.
          )pydoc");
    m.def("save",
          &save,
          py::arg("data"),
          py::arg("filename"),
          py::arg("mode") = "w",
          R"pydoc(
          Save `data` to file with given `filename`.

          This function guesses the type of the file from the filename ending.
          )pydoc");
    m.def("load",
          &load,
          py::arg("filename"),
          R"pydoc(
          Load data from file with given `filename`.
          )pydoc");

    py::class_<Hdf5Exportable, PyHdf5Exportable, py::smart_holder> exportable(
      m, "Hdf5Exportable", py::dynamic_attr());
    exportable.doc() = R"pydoc(
        Interface specification for a class to be exportable to our HDF5 format.

        To allow a class to be exported to HDF5 with :func:`save_to_hdf5`,
        it only needs to implement the :meth:`save_hdf5` method as documented below.
        To allow import, a class should implement the classmethod :meth:`from_hdf5`.
        )pydoc";
    exportable.def(py::init<>())
      .def("save_hdf5",
           &Hdf5Exportable::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static(
        "from_hdf5",
        [](py::handle cls,
           py::object hdf5_loader,
           py::object h5gr,
           std::string subpath) {
            return Hdf5Exportable::from_hdf5(
              cls.cast<py::type>(), std::move(hdf5_loader), std::move(h5gr), std::move(subpath));
        },
        py::is_method(exportable),
        py::arg("hdf5_loader"),
        py::arg("h5gr"),
        py::arg("subpath"));

    py::class_<Hdf5Ignored>(m, "Hdf5Ignored")
      .def(py::init<std::string>(), py::arg("name") = "unknown")
      .def_readwrite("name", &Hdf5Ignored::name);

    py::class_<Hdf5Saver>(m, "Hdf5Saver")
      .def(py::init<py::object, py::object>(),
           py::arg("h5group"),
           py::arg("format_selection") = py::none())
      .def_readwrite("h5group", &Hdf5Saver::h5group)
      .def_readwrite("memo_save", &Hdf5Saver::memo_save)
      .def_readwrite("format_selection", &Hdf5Saver::format_selection)
      .def("save", &Hdf5Saver::save, py::arg("obj"), py::arg("path") = "/")
      .def("create_group_for_obj",
           &Hdf5Saver::create_group_for_obj,
           py::arg("path"),
           py::arg("obj"))
      .def("memorize_save", &Hdf5Saver::memorize_save, py::arg("h5gr"), py::arg("obj"))
      .def("save_reduce",
           &Hdf5Saver::save_reduce,
           py::arg("func"),
           py::arg("args"),
           py::arg("state") = py::none(),
           py::arg("listitems") = py::none(),
           py::arg("dictitems") = py::none(),
           py::arg("state_setter") = py::none(),
           py::arg("obj") = py::none(),
           py::arg("path") = py::none())
      .def("save_none", &Hdf5Saver::save_none, py::arg("obj"), py::arg("path"), py::arg("type_repr"))
      .def("save_dataset",
           &Hdf5Saver::save_dataset,
           py::arg("obj"),
           py::arg("path"),
           py::arg("type_repr"))
      .def("save_masked_array",
           &Hdf5Saver::save_masked_array,
           py::arg("obj"),
           py::arg("path"),
           py::arg("type_repr"))
      .def("save_iterable",
           &Hdf5Saver::save_iterable,
           py::arg("obj"),
           py::arg("path"),
           py::arg("type_repr"))
      .def("save_iterable_content",
           &Hdf5Saver::save_iterable_content,
           py::arg("obj"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def("save_dict", &Hdf5Saver::save_dict, py::arg("obj"), py::arg("path"), py::arg("type_repr"))
      .def("save_dict_content",
           &Hdf5Saver::save_dict_content,
           py::arg("obj"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def("save_range",
           &Hdf5Saver::save_range,
           py::arg("obj"),
           py::arg("path"),
           py::arg("type_repr"))
      .def("save_dtype",
           &Hdf5Saver::save_dtype,
           py::arg("obj"),
           py::arg("path"),
           py::arg("type_repr"))
      .def("save_ignored",
           &Hdf5Saver::save_ignored,
           py::arg("obj"),
           py::arg("path"),
           py::arg("type_repr"))
      .def("save_global",
           &Hdf5Saver::save_global,
           py::arg("obj"),
           py::arg("path"),
           py::arg("type_repr"));

    py::class_<Hdf5Loader>(m, "Hdf5Loader")
      .def(py::init<py::object, bool, py::object>(),
           py::arg("h5group"),
           py::arg("ignore_unknown") = true,
           py::arg("exclude") = py::none())
      .def_readwrite("h5group", &Hdf5Loader::h5group)
      .def_readwrite("ignore_unknown", &Hdf5Loader::ignore_unknown)
      .def_readwrite("memo_load", &Hdf5Loader::memo_load)
      .def("load", &Hdf5Loader::load, py::arg("path") = py::none())
      .def("memorize_load", &Hdf5Loader::memorize_load, py::arg("h5gr"), py::arg("obj"))
      .def("get_all_hdf5_keys", &Hdf5Loader::get_all_hdf5_keys, py::arg("h5_group") = py::none())
      .def_static("get_attr", &Hdf5Loader::get_attr, py::arg("h5gr"), py::arg("attr_name"))
      .def("load_none",
           &Hdf5Loader::load_none,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_dataset",
           &Hdf5Loader::load_dataset,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_str",
           &Hdf5Loader::load_str,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_converted_to_str",
           &Hdf5Loader::load_converted_to_str,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_masked_array",
           &Hdf5Loader::load_masked_array,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_list",
           &Hdf5Loader::load_list,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_set",
           &Hdf5Loader::load_set,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_tuple",
           &Hdf5Loader::load_tuple,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_dict",
           &Hdf5Loader::load_dict,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_general_dict",
           &Hdf5Loader::load_general_dict,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_simple_dict",
           &Hdf5Loader::load_simple_dict,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_range",
           &Hdf5Loader::load_range,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_dtype",
           &Hdf5Loader::load_dtype,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_hdf5exportable",
           &Hdf5Loader::load_hdf5exportable,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_ignored",
           &Hdf5Loader::load_ignored,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_global",
           &Hdf5Loader::load_global,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"))
      .def("load_reduce",
           &Hdf5Loader::load_reduce,
           py::arg("h5gr"),
           py::arg("type_info"),
           py::arg("subpath"));

    m.def("save_to_hdf5",
          &save_to_hdf5,
          py::arg("h5group"),
          py::arg("obj"),
          py::arg("path") = "/",
          R"pydoc(
          Save an object `obj` into a hdf5 file or group.
          )pydoc");
    m.def("load_from_hdf5",
          &load_from_hdf5,
          py::arg("h5group"),
          py::arg("path") = py::none(),
          py::arg("ignore_unknown") = true,
          py::arg("exclude") = py::none(),
          R"pydoc(
          Load an object from hdf5 file or group.
          )pydoc");

    init_dispatch_tables(m);
}

} // namespace hdf5_io
