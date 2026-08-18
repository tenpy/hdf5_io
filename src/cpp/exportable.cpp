#include <hdf5_io/h5_bridge.h>
#include <hdf5_io/hdf5_io.h>

namespace hdf5_io {

Hdf5Ignored::Hdf5Ignored(std::string name_)
  : name(std::move(name_))
{
}

void
Hdf5Exportable::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string subpath)
{
    py::object self = py::cast(this);
    py::object type_repr =
      hdf5_saver.attr("save_dict_content")(self.attr("__dict__"), h5gr, subpath);
    h5_set_attr(h5gr, ATTR_FORMAT, type_repr);
}

py::object
Hdf5Exportable::from_hdf5(py::type cls,
                          py::object hdf5_loader,
                          py::object h5gr,
                          std::string subpath)
{
    py::object dict_format = hdf5_loader.attr("get_attr")(h5gr, ATTR_FORMAT);
    py::object obj = cls.attr("__new__")(cls);
    hdf5_loader.attr("memorize_load")(h5gr, obj);
    py::object data = hdf5_loader.attr("load_dict")(h5gr, dict_format, subpath);
    obj.attr("__dict__").attr("update")(data);
    return obj;
}

} // namespace hdf5_io
