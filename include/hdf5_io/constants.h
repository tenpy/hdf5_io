#pragma once

#include <string>

namespace hdf5_io {

inline constexpr char const* REPR_IGNORED = "ignore";
inline constexpr char const* REPR_HDF5EXPORTABLE = "instance";
inline constexpr char const* REPR_REDUCE = "reduce";
inline constexpr char const* REPR_ARRAY = "array";
inline constexpr char const* REPR_MASKED_ARRAY = "masked_array";
inline constexpr char const* REPR_INT = "int";
inline constexpr char const* REPR_INT_AS_STR = "int_as_str";
inline constexpr char const* REPR_FLOAT = "float";
inline constexpr char const* REPR_STR = "str";
inline constexpr char const* REPR_BYTES = "bytes";
inline constexpr char const* REPR_COMPLEX = "complex";
inline constexpr char const* REPR_INT64 = "np.int64";
inline constexpr char const* REPR_FLOAT64 = "np.float64";
inline constexpr char const* REPR_COMPLEX128 = "np.complex128";
inline constexpr char const* REPR_INT32 = "np.int32";
inline constexpr char const* REPR_FLOAT32 = "np.float32";
inline constexpr char const* REPR_COMPLEX64 = "np.complex64";
inline constexpr char const* REPR_BOOL = "bool";
inline constexpr char const* REPR_NONE = "None";
inline constexpr char const* REPR_RANGE = "range";
inline constexpr char const* REPR_LIST = "list";
inline constexpr char const* REPR_TUPLE = "tuple";
inline constexpr char const* REPR_SET = "set";
inline constexpr char const* REPR_DICT_GENERAL = "dict";
inline constexpr char const* REPR_DICT_SIMPLE = "simple_dict";
inline constexpr char const* REPR_DTYPE = "dtype";
inline constexpr char const* REPR_FUNCTION = "function";
inline constexpr char const* REPR_CLASS = "class";
inline constexpr char const* REPR_GLOBAL = "global";

inline constexpr char const* ATTR_TYPE = "type";
inline constexpr char const* ATTR_CLASS = "class";
inline constexpr char const* ATTR_MODULE = "module";
inline constexpr char const* ATTR_LEN = "len";
inline constexpr char const* ATTR_FORMAT = "format";

} // namespace hdf5_io
