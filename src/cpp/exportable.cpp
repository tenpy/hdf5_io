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
    Hdf5Saver& saver = hdf5_saver.cast<Hdf5Saver&>();
    auto group = wrap_group(h5gr);
    py::object self = py::cast(this);
    std::string type_repr = saver.save_dict_content(self.attr("__dict__"), group, subpath);
    h5_set_attr(hid_from_h5py(h5gr), ATTR_FORMAT, type_repr);
}

py::object
Hdf5Exportable::from_hdf5(py::type cls,
                          py::object hdf5_loader,
                          py::object h5gr,
                          std::string subpath)
{
    Hdf5Loader& loader = hdf5_loader.cast<Hdf5Loader&>();
    hid_t hid = hid_from_h5py(h5gr);
    std::string dict_format = py::str(loader.get_attr(hid, ATTR_FORMAT)).cast<std::string>();
    py::object obj = cls.attr("__new__")(cls);
    loader.memorize_load(hid, obj);
    py::object data = loader.load_dict(hid, dict_format, subpath);
    obj.attr("__dict__").attr("update")(data);
    return obj;
}

} // namespace hdf5_io
