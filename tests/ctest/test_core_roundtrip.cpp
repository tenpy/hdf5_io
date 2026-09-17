#include <hdf5_io/core_io.h>
#include <hdf5_io/h5_ops.h>

#include <highfive/highfive.hpp>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int
main()
{
    fs::path tmp = fs::temp_directory_path() / "hdf5_io_core_roundtrip.h5";
    {
        HighFive::File file(tmp.string(), HighFive::File::Truncate);
        hdf5_io::core::Saver saver(file);

        saver.save_none("none_val");
        saver.save_int64("answer", 42);
        saver.save_string("hello", "world");

        HighFive::Group list_g;
        std::string list_sub;
        saver.save_sequence_begin("mylist", hdf5_io::REPR_LIST, 2, list_g, list_sub);
        // Children under root path mylist/0 — use a nested Saver rooted at list_g
        hdf5_io::core::Saver list_saver(list_g);
        list_saver.save_int64("0", 1);
        list_saver.save_int64("1", 2);

        HighFive::Group dict_g;
        std::string dict_sub;
        saver.save_simple_dict_begin("mydict", dict_g, dict_sub);
        hdf5_io::core::Saver dict_saver(dict_g);
        dict_saver.save_string("key", "value");

        hdf5_io::Hdf5Buffer buf;
        buf.kind = 'f';
        buf.itemsize = 8;
        buf.shape = {2};
        buf.storage = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[16]);
        double* d = reinterpret_cast<double*>(buf.storage.get());
        d[0] = 1.5;
        d[1] = 2.5;
        saver.save_array("arr", buf, hdf5_io::REPR_ARRAY);

        // Hard-link sharing: save exportable-like group twice via memo
        HighFive::Group g1 = hdf5_io::h5_create_group(saver.root(), "obj");
        hdf5_io::h5_set_attr(g1.getId(), hdf5_io::ATTR_TYPE, std::string(hdf5_io::REPR_HDF5EXPORTABLE));
        int dummy = 0;
        saver.memorize_save(&dummy, g1.getId());
        hid_t existing = -1;
        assert(saver.already_saved(&dummy, existing));
        hdf5_io::h5_hard_link(saver.root(), "obj_alias", existing);
    }

    {
        HighFive::File file(tmp.string(), HighFive::File::ReadOnly);
        hdf5_io::core::Loader loader(file);

        hid_t none_id = loader.open("none_val");
        assert(loader.type_of(none_id) == hdf5_io::REPR_NONE);
        H5Idec_ref(none_id);

        hid_t ans = loader.open("answer");
        assert(loader.load_int64(ans) == 42);
        H5Idec_ref(ans);

        hid_t hello = loader.open("hello");
        assert(loader.load_string(hello) == "world");
        H5Idec_ref(hello);

        hid_t arr = loader.open("arr");
        auto buf = loader.load_array(arr);
        assert(buf.shape.size() == 1 && buf.shape[0] == 2);
        double const* d = reinterpret_cast<double const*>(buf.data());
        assert(d[0] == 1.5 && d[1] == 2.5);
        H5Idec_ref(arr);

        auto root = file.getGroup("/");
        assert(hdf5_io::h5_contains(root, "obj_alias"));
        hid_t a = loader.open("obj");
        hid_t b = loader.open("obj_alias");
        assert(hdf5_io::object_token(a) == hdf5_io::object_token(b));
        H5Idec_ref(a);
        H5Idec_ref(b);
    }

    fs::remove(tmp);
    std::cout << "hdf5_io core roundtrip OK\n";
    return 0;
}
