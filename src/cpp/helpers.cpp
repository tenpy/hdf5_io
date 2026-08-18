#include <hdf5_io/hdf5_io.h>

#include <pybind11/pybind11.h>

namespace hdf5_io {

bool
valid_hdf5_path_component(py::handle name)
{
    if (!py::isinstance<py::str>(name))
        return false;
    std::string s = name.cast<std::string>();
    return s.find('/') == std::string::npos && s != ".";
}

py::object
find_global(std::string const& module, std::string const& qualified_name)
{
    py::object obj = py::module_::import("importlib").attr("import_module")(module);
    py::module_ builtins = py::module_::import("builtins");
    for (auto const& part : py::str(qualified_name).attr("split")(".")) {
        obj = builtins.attr("getattr")(obj, part);
    }
    return obj;
}

} // namespace hdf5_io
