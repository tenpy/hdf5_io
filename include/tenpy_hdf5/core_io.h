#pragma once

/// Python-free TeNPy-format HDF5 saver/loader for linking into C++ libraries (e.g. cyten).

#include <tenpy_hdf5/constants.h>
#include <tenpy_hdf5/exceptions.h>
#include <tenpy_hdf5/h5_ops.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <highfive/highfive.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace tenpy_hdf5::core {

class Saver;
class Loader;

/// Opaque loaded value used when the concrete C++ type is not known statically.
struct Value;

using SaveFn = std::function<void(Saver&, HighFive::Group&, std::string const& subpath, void*)>;
using LoadFn = std::function<std::shared_ptr<void>(Loader&, HighFive::Group&, std::string const& subpath)>;

void register_type(std::string module, std::string class_name, SaveFn save, LoadFn load);
bool has_registered_type(std::string const& module, std::string const& class_name);

class Saver
{
  public:
    explicit Saver(HighFive::Group root);
    explicit Saver(HighFive::File& file);

    HighFive::Group& root() { return root_; }

    /// Memoize by C++ object identity; returns existing hid if already saved (hard-link target).
    hid_t memorize_save(void const* identity, hid_t h5obj);
    bool already_saved(void const* identity, hid_t& out_hid) const;

    std::pair<HighFive::Group, std::string> create_group_for_obj(std::string const& path,
                                                                void const* identity);

    hid_t save_none(std::string const& path);
    hid_t save_bool(std::string const& path, bool v);
    hid_t save_int64(std::string const& path, std::int64_t v);
    hid_t save_float64(std::string const& path, double v);
    hid_t save_complex(std::string const& path, std::complex<double> const& v);
    hid_t save_string(std::string const& path, std::string const& v);
    hid_t save_bytes(std::string const& path, std::string const& v);
    hid_t save_array(std::string const& path, Hdf5Buffer const& buf, char const* type_repr = REPR_ARRAY);

    /// Create a list/tuple/set group; caller fills children ``0..n-1``.
    hid_t save_sequence_begin(std::string const& path,
                              char const* type_repr,
                              std::int64_t len,
                              HighFive::Group& out_group,
                              std::string& out_subpath);

    /// Create a simple_dict group; caller fills children named by keys.
    hid_t save_simple_dict_begin(std::string const& path,
                                 HighFive::Group& out_group,
                                 std::string& out_subpath);

    /// Save a registered exportable instance (writes type/module/class attrs, then calls registry).
    hid_t save_exportable(std::string const& path,
                          std::string const& module,
                          std::string const& class_name,
                          void* obj,
                          void const* identity);

    /// Write an instance group and run `body` (no registry required for save).
    hid_t save_instance(std::string const& path,
                        std::string const& module,
                        std::string const& class_name,
                        void const* identity,
                        std::function<void(Saver&, HighFive::Group&, std::string const&)> body);

    void set_type_attr(hid_t loc, char const* type_repr);

  private:
    HighFive::Group root_;
    std::unordered_map<std::uintptr_t, hid_t> memo_;
};

class Loader
{
  public:
    explicit Loader(HighFive::Group root, bool ignore_unknown = true);
    explicit Loader(HighFive::File& file, bool ignore_unknown = true);

    HighFive::Group& root() { return root_; }
    bool ignore_unknown() const { return ignore_unknown_; }

    void memorize_load(hid_t h5obj, std::shared_ptr<void> obj);
    void memorize_load(hid_t h5obj, void* raw); // non-owning raw pointer stored as shared_ptr with no-op deleter
    std::shared_ptr<void> lookup_memo(hid_t h5obj) const;

    std::string type_of(hid_t h5obj) const;

    hid_t open(std::string const& path);

    void load_none(hid_t h5obj);
    bool load_bool(hid_t h5obj);
    std::int64_t load_int64(hid_t h5obj);
    double load_float64(hid_t h5obj);
    std::complex<double> load_complex(hid_t h5obj);
    std::string load_string(hid_t h5obj);
    Hdf5Buffer load_array(hid_t h5obj);

    std::shared_ptr<void> load_exportable(hid_t h5obj, std::string const& subpath);

    /// Dispatch by ATTR_TYPE; for instance uses registry. Returns type string.
    std::string load_dispatch(hid_t h5obj, std::string const& subpath, std::shared_ptr<void>& out);

  private:
    HighFive::Group root_;
    bool ignore_unknown_ = true;
    std::unordered_map<std::string, std::shared_ptr<void>> memo_;
};

} // namespace tenpy_hdf5::core
