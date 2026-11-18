#include <hdf5_io/hdf5_io.h>

namespace hdf5_io {

bool
valid_hdf5_path_component(py::handle name)
{
    if (!py::isinstance<py::str>(name))
        return false;
    std::string s = py::cast<std::string>(name);
    return s.find('/') == std::string::npos && s != ".";
}

py::object
find_global(std::string const& module, std::string const& qualified_name)
{
    py::object obj = py::module_::import_("importlib").attr("import_module")(module);
    py::module_ builtins = py::module_::import_("builtins");
    for (auto const& part : to_str(qualified_name).attr("split")(".")) {
        obj = builtins.attr("getattr")(obj, part);
    }
    return obj;
}

} // namespace hdf5_io
