#include <tenpy_hdf5/core_io.h>

#include <hdf5.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace tenpy_hdf5::core {

namespace {

struct RegistryEntry
{
    SaveFn save;
    LoadFn load;
};

using RegistryKey = std::pair<std::string, std::string>;

struct RegistryKeyHash
{
    std::size_t operator()(RegistryKey const& k) const
    {
        return std::hash<std::string>{}(k.first) ^ (std::hash<std::string>{}(k.second) << 1);
    }
};

std::unordered_map<RegistryKey, RegistryEntry, RegistryKeyHash>&
registry()
{
    static std::unordered_map<RegistryKey, RegistryEntry, RegistryKeyHash> r;
    return r;
}

std::mutex&
registry_mutex()
{
    static std::mutex m;
    return m;
}

std::string
ensure_trailing_slash(std::string path)
{
    if (path.empty() || path.back() != '/')
        path.push_back('/');
    return path;
}

} // namespace

void
register_type(std::string module, std::string class_name, SaveFn save, LoadFn load)
{
    std::lock_guard lock(registry_mutex());
    registry()[{std::move(module), std::move(class_name)}] =
      RegistryEntry{std::move(save), std::move(load)};
}

bool
has_registered_type(std::string const& module, std::string const& class_name)
{
    std::lock_guard lock(registry_mutex());
    return registry().contains({module, class_name});
}

Saver::Saver(HighFive::Group root)
  : root_(std::move(root))
{}

Saver::Saver(HighFive::File& file)
  : root_(file.getGroup("/"))
{}

hid_t
Saver::memorize_save(void const* identity, hid_t h5obj)
{
    auto key = reinterpret_cast<std::uintptr_t>(identity);
    auto it = memo_.find(key);
    if (it != memo_.end())
        return it->second;
    check_hdf5(H5Iinc_ref(h5obj), "H5Iinc_ref");
    memo_[key] = h5obj;
    return h5obj;
}

bool
Saver::already_saved(void const* identity, hid_t& out_hid) const
{
    auto it = memo_.find(reinterpret_cast<std::uintptr_t>(identity));
    if (it == memo_.end())
        return false;
    out_hid = it->second;
    return true;
}

std::pair<HighFive::Group, std::string>
Saver::create_group_for_obj(std::string const& path, void const* identity)
{
    hid_t existing = -1;
    if (identity && already_saved(identity, existing)) {
        HighFive::Group g = group_from_hid(existing);
        check_hdf5(H5Iinc_ref(existing), "H5Iinc_ref");
        return {std::move(g), ensure_trailing_slash(h5_object_name(existing))};
    }
    if (path == "/" || path.empty()) {
        check_hdf5(H5Iinc_ref(root_.getId()), "H5Iinc_ref");
        return {group_from_hid(root_.getId()), "/"};
    }
    HighFive::Group g = h5_create_group(root_, path);
    return {std::move(g), ensure_trailing_slash(path)};
}

void
Saver::set_type_attr(hid_t loc, char const* type_repr)
{
    h5_set_attr(loc, ATTR_TYPE, std::string(type_repr));
}

hid_t
Saver::save_none(std::string const& path)
{
    h5_write_dataset(root_, path, std::string("None"));
    hid_t id = h5_open(root_, path);
    set_type_attr(id, REPR_NONE);
    return id;
}

hid_t
Saver::save_bool(std::string const& path, bool v)
{
    h5_write_dataset(root_, path, v);
    hid_t id = h5_open(root_, path);
    set_type_attr(id, REPR_BOOL);
    return id;
}

hid_t
Saver::save_int64(std::string const& path, std::int64_t v)
{
    h5_write_dataset(root_, path, v);
    hid_t id = h5_open(root_, path);
    set_type_attr(id, REPR_INT);
    return id;
}

hid_t
Saver::save_float64(std::string const& path, double v)
{
    h5_write_dataset(root_, path, v);
    hid_t id = h5_open(root_, path);
    set_type_attr(id, REPR_FLOAT);
    return id;
}

hid_t
Saver::save_complex(std::string const& path, std::complex<double> const& v)
{
    h5_write_dataset(root_, path, v);
    hid_t id = h5_open(root_, path);
    set_type_attr(id, REPR_COMPLEX);
    return id;
}

hid_t
Saver::save_string(std::string const& path, std::string const& v)
{
    h5_write_dataset(root_, path, v);
    hid_t id = h5_open(root_, path);
    set_type_attr(id, REPR_STR);
    return id;
}

hid_t
Saver::save_bytes(std::string const& path, std::string const& v)
{
    // Store as ASCII vlen string dataset with REPR_BYTES
    h5_write_dataset(root_, path, v);
    hid_t id = h5_open(root_, path);
    set_type_attr(id, REPR_BYTES);
    return id;
}

hid_t
Saver::save_array(std::string const& path, Hdf5Buffer const& buf, char const* type_repr)
{
    h5_write_dataset(root_, path, buf);
    hid_t id = h5_open(root_, path);
    set_type_attr(id, type_repr);
    return id;
}

hid_t
Saver::save_sequence_begin(std::string const& path,
                           char const* type_repr,
                           std::int64_t len,
                           HighFive::Group& out_group,
                           std::string& out_subpath)
{
    out_group = h5_create_group(root_, path);
    out_subpath = ensure_trailing_slash(path);
    set_type_attr(out_group.getId(), type_repr);
    h5_set_attr(out_group.getId(), ATTR_LEN, len);
    check_hdf5(H5Iinc_ref(out_group.getId()), "H5Iinc_ref");
    return out_group.getId();
}

hid_t
Saver::save_simple_dict_begin(std::string const& path,
                              HighFive::Group& out_group,
                              std::string& out_subpath)
{
    out_group = h5_create_group(root_, path);
    out_subpath = ensure_trailing_slash(path);
    set_type_attr(out_group.getId(), REPR_DICT_SIMPLE);
    check_hdf5(H5Iinc_ref(out_group.getId()), "H5Iinc_ref");
    return out_group.getId();
}

hid_t
Saver::save_exportable(std::string const& path,
                       std::string const& module,
                       std::string const& class_name,
                       void* obj,
                       void const* identity)
{
    SaveFn save_fn;
    {
        std::lock_guard lock(registry_mutex());
        auto it = registry().find({module, class_name});
        if (it == registry().end())
            throw Hdf5ExportError("no registered save for " + module + "." + class_name);
        save_fn = it->second.save;
    }
    return save_instance(path, module, class_name, identity, [&](Saver& s, HighFive::Group& g, std::string const& sub) {
        save_fn(s, g, sub, obj);
    });
}

hid_t
Saver::save_instance(std::string const& path,
                     std::string const& module,
                     std::string const& class_name,
                     void const* identity,
                     std::function<void(Saver&, HighFive::Group&, std::string const&)> body)
{
    hid_t existing = -1;
    if (identity && already_saved(identity, existing)) {
        h5_hard_link(root_, path, existing);
        check_hdf5(H5Iinc_ref(existing), "H5Iinc_ref");
        return existing;
    }
    auto [group, subpath] = create_group_for_obj(path, identity);
    set_type_attr(group.getId(), REPR_HDF5EXPORTABLE);
    h5_set_attr(group.getId(), ATTR_MODULE, module);
    h5_set_attr(group.getId(), ATTR_CLASS, class_name);
    if (identity)
        memorize_save(identity, group.getId());
    body(*this, group, subpath);
    check_hdf5(H5Iinc_ref(group.getId()), "H5Iinc_ref");
    return group.getId();
}

Loader::Loader(HighFive::Group root, bool ignore_unknown)
  : root_(std::move(root))
  , ignore_unknown_(ignore_unknown)
{}

Loader::Loader(HighFive::File& file, bool ignore_unknown)
  : root_(file.getGroup("/"))
  , ignore_unknown_(ignore_unknown)
{}

void
Loader::memorize_load(hid_t h5obj, std::shared_ptr<void> obj)
{
    memo_[object_token(h5obj)] = std::move(obj);
}

void
Loader::memorize_load(hid_t h5obj, void* raw)
{
    memo_[object_token(h5obj)] = std::shared_ptr<void>(raw, [](void*) {});
}

std::shared_ptr<void>
Loader::lookup_memo(hid_t h5obj) const
{
    auto it = memo_.find(object_token(h5obj));
    if (it == memo_.end())
        return nullptr;
    return it->second;
}

std::string
Loader::type_of(hid_t h5obj) const
{
    auto t = h5_get_attr_string(h5obj, ATTR_TYPE);
    if (!t)
        throw Hdf5ImportError("missing type attribute");
    return *t;
}

hid_t
Loader::open(std::string const& path)
{
    return h5_open(root_, path);
}

void
Loader::load_none(hid_t /*h5obj*/)
{}

bool
Loader::load_bool(hid_t h5obj)
{
    auto buf = h5_read_buffer(h5obj);
    return buf.storage && buf.storage[0] != 0;
}

std::int64_t
Loader::load_int64(hid_t h5obj)
{
    auto buf = h5_read_buffer(h5obj);
    if (buf.kind == 'i' && buf.itemsize == 8)
        return *reinterpret_cast<std::int64_t const*>(buf.data());
    if (buf.kind == 'i' && buf.itemsize == 4)
        return *reinterpret_cast<std::int32_t const*>(buf.data());
    throw Hdf5ImportError("expected integer dataset");
}

double
Loader::load_float64(hid_t h5obj)
{
    auto buf = h5_read_buffer(h5obj);
    if (buf.kind == 'f' && buf.itemsize == 8)
        return *reinterpret_cast<double const*>(buf.data());
    if (buf.kind == 'f' && buf.itemsize == 4)
        return *reinterpret_cast<float const*>(buf.data());
    throw Hdf5ImportError("expected float dataset");
}

std::complex<double>
Loader::load_complex(hid_t h5obj)
{
    auto buf = h5_read_buffer(h5obj);
    if (buf.kind == 'c' && buf.itemsize == 16)
        return *reinterpret_cast<std::complex<double> const*>(buf.data());
    throw Hdf5ImportError("expected complex dataset");
}

std::string
Loader::load_string(hid_t h5obj)
{
    return h5_read_vlen_string(h5obj);
}

Hdf5Buffer
Loader::load_array(hid_t h5obj)
{
    return h5_read_buffer(h5obj);
}

std::shared_ptr<void>
Loader::load_exportable(hid_t h5obj, std::string const& subpath)
{
    if (auto existing = lookup_memo(h5obj))
        return existing;
    auto module = h5_get_attr_string(h5obj, ATTR_MODULE);
    auto cls = h5_get_attr_string(h5obj, ATTR_CLASS);
    if (!module || !cls)
        throw Hdf5ImportError("instance missing module/class attrs");
    LoadFn load_fn;
    {
        std::lock_guard lock(registry_mutex());
        auto it = registry().find({*module, *cls});
        if (it == registry().end()) {
            if (ignore_unknown_)
                return nullptr;
            throw Hdf5ImportError("no registered load for " + *module + "." + *cls);
        }
        load_fn = it->second.load;
    }
    HighFive::Group group = group_from_hid(h5obj);
    check_hdf5(H5Iinc_ref(h5obj), "H5Iinc_ref");
    auto obj = load_fn(*this, group, ensure_trailing_slash(subpath));
    memorize_load(h5obj, obj);
    return obj;
}

std::string
Loader::load_dispatch(hid_t h5obj, std::string const& subpath, std::shared_ptr<void>& out)
{
    if (auto existing = lookup_memo(h5obj)) {
        out = existing;
        return type_of(h5obj);
    }
    std::string t = type_of(h5obj);
    if (t == REPR_HDF5EXPORTABLE) {
        out = load_exportable(h5obj, subpath);
        return t;
    }
    out = nullptr;
    return t;
}

} // namespace tenpy_hdf5::core
