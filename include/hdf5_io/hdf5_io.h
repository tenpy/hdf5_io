#pragma once

#include <hdf5_io/constants.h>
#include <hdf5_io/exceptions.h>
#include <pybind11/pybind11.h>
#include <string>
#include <utility>

namespace hdf5_io {

namespace py = pybind11;

bool valid_hdf5_path_component(py::handle name);
py::object find_global(std::string const& module, std::string const& qualified_name);

void save(py::object data, std::string filename, std::string mode = "w");
py::object load(std::string filename);

class Hdf5Exportable
{
  public:
    Hdf5Exportable() = default;
    virtual ~Hdf5Exportable() = default;

    virtual void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string subpath);

    static py::object from_hdf5(py::type cls,
                                py::object hdf5_loader,
                                py::object h5gr,
                                std::string subpath);
};

class Hdf5Ignored
{
  public:
    std::string name;
    explicit Hdf5Ignored(std::string name = "unknown");
};

class Hdf5Saver
{
  public:
    py::object h5group;
    py::dict memo_save;
    py::dict format_selection;

    explicit Hdf5Saver(py::object h5group, py::object format_selection = py::none());

    py::object save(py::object obj, std::string path = "/");
    std::pair<py::object, std::string> create_group_for_obj(std::string const& path,
                                                            py::object obj);
    void memorize_save(py::object h5gr, py::object obj);

    py::object save_reduce(py::object func,
                           py::object args,
                           py::object state = py::none(),
                           py::object listitems = py::none(),
                           py::object dictitems = py::none(),
                           py::object state_setter = py::none(),
                           py::object obj = py::none(),
                           py::object path = py::none());

    py::object save_none(py::object obj, std::string const& path, std::string const& type_repr);
    py::object save_dataset(py::object obj, std::string const& path, std::string type_repr);
    py::object save_masked_array(py::object obj,
                                 std::string const& path,
                                 std::string const& type_repr);
    py::object save_iterable(py::object obj, std::string const& path, std::string const& type_repr);
    void save_iterable_content(py::object obj, py::object h5gr, std::string const& subpath);
    py::object save_dict(py::object obj, std::string const& path, std::string const& type_repr);
    std::string save_dict_content(py::object obj, py::object h5gr, std::string const& subpath);
    py::object save_range(py::object obj, std::string const& path, std::string const& type_repr);
    py::object save_dtype(py::object obj, std::string const& path, std::string const& type_repr);
    py::object save_ignored(py::object obj, std::string const& path, std::string const& type_repr);
    py::object save_global(py::object obj, std::string const& path, std::string const& type_repr);

  private:
    py::dict dispatch_save() const;
};

class Hdf5Loader
{
  public:
    py::object h5group;
    bool ignore_unknown = true;
    py::dict memo_load;

    explicit Hdf5Loader(py::object h5group,
                        bool ignore_unknown = true,
                        py::object exclude = py::none());

    py::object load(py::object path = py::none());
    void memorize_load(py::object h5gr, py::object obj);
    py::object get_all_hdf5_keys(py::object h5_group = py::none());
    static py::object get_attr(py::object h5gr, std::string const& attr_name);

    py::object load_none(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_dataset(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_str(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_converted_to_str(py::object h5gr,
                                     py::object type_info,
                                     std::string const& subpath);
    py::object load_masked_array(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_list(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_set(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_tuple(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_dict(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_general_dict(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_simple_dict(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_range(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_dtype(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_hdf5exportable(py::object h5gr,
                                   py::object type_info,
                                   std::string const& subpath);
    py::object load_ignored(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_global(py::object h5gr, py::object type_info, std::string const& subpath);
    py::object load_reduce(py::object h5gr, py::object type_info, std::string const& subpath);

  private:
    py::dict dispatch_load() const;
};

py::object save_to_hdf5(py::object h5group, py::object obj, std::string path = "/");
py::object load_from_hdf5(py::object h5group,
                          py::object path = py::none(),
                          bool ignore_unknown = true,
                          py::object exclude = py::none());

/// Fill Hdf5Saver.dispatch_save and Hdf5Loader.dispatch_load after pybind class registration.
void init_dispatch_tables(py::module_& m);

} // namespace hdf5_io
