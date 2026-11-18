#pragma once

/// Python/nanobind glue shared by the Python-facing part of tenpy_hdf5.
///
/// This header must never be included from the Python-free ``tenpy_hdf5_core`` sources.
/// The module is built against the Python stable ABI (``Py_LIMITED_API``), so numpy is only
/// accessed through its Python-level API and the buffer protocol (no NumPy C API).

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <string>
#include <vector>

namespace tenpy_hdf5 {

namespace py = nanobind;

/// ``str`` from a std::string (handles embedded NUL / non-NUL-terminated data).
inline py::str
to_str(std::string const& s)
{
    return py::str(s.data(), s.size());
}

/// ``bytes`` from a std::string.
inline py::bytes
to_bytes(std::string const& s)
{
    return py::bytes(s.data(), s.size());
}

/// Python truthiness (``bool(h)``); unlike ``py::cast<bool>`` this also accepts ``numpy.bool_``.
inline bool
truthy(py::handle h)
{
    int r = PyObject_IsTrue(h.ptr());
    if (r < 0)
        py::raise_python_error();
    return r == 1;
}

/// ``numpy.dtype`` wrapper (only what the HDF5 bridge needs).
class NumpyDtype : public py::object
{
    static bool check(PyObject* o)
    {
        auto np = py::module_::import_("numpy");
        int r = PyObject_IsInstance(o, np.attr("dtype").ptr());
        if (r < 0)
            PyErr_Clear();
        return r == 1;
    }

  public:
    NB_OBJECT_DEFAULT(NumpyDtype, object, "numpy.dtype", check)

    /// ``numpy.dtype(spec)`` for a string spec such as ``"uint8"``.
    static NumpyDtype from_spec(char const* spec)
    {
        return py::steal<NumpyDtype>(
          py::module_::import_("numpy").attr("dtype")(spec).release());
    }

    /// Single character code (``'f'``, ``'c'``, ``'i'``, ``'u'``, ``'b'``, ...).
    char kind() const
    {
        std::string k = py::cast<std::string>(attr("kind"));
        return k.empty() ? '\0' : k[0];
    }

    Py_ssize_t itemsize() const { return py::cast<Py_ssize_t>(attr("itemsize")); }

    std::string name() const { return py::cast<std::string>(attr("name")); }
};

/// ``numpy.ndarray`` wrapper. Data access goes through the buffer protocol.
class NumpyArray : public py::object
{
    static bool check(PyObject* o)
    {
        auto np = py::module_::import_("numpy");
        int r = PyObject_IsInstance(o, np.attr("ndarray").ptr());
        if (r < 0)
            PyErr_Clear();
        return r == 1;
    }

    static void* buffer_ptr(PyObject* o, int flags)
    {
        Py_buffer view;
        if (PyObject_GetBuffer(o, &view, flags) != 0)
            py::raise_python_error();
        void* ptr = view.buf;
        PyBuffer_Release(&view);
        return ptr;
    }

  public:
    NB_OBJECT_DEFAULT(NumpyArray, object, "numpy.ndarray", check)

    /// ``numpy.asarray(obj)``
    static NumpyArray from_object(py::handle obj)
    {
        return py::steal<NumpyArray>(
          py::module_::import_("numpy").attr("asarray")(obj).release());
    }

    /// ``numpy.empty(shape, dtype)``
    static NumpyArray empty(NumpyDtype const& dtype, std::vector<Py_ssize_t> const& shape)
    {
        py::list dims;
        for (Py_ssize_t d : shape)
            dims.append(d);
        return py::steal<NumpyArray>(py::module_::import_("numpy")
                                       .attr("empty")(py::tuple(dims), dtype)
                                       .release());
    }

    Py_ssize_t ndim() const { return py::cast<Py_ssize_t>(attr("ndim")); }

    Py_ssize_t shape(Py_ssize_t i) const
    {
        return py::cast<Py_ssize_t>(attr("shape")[py::int_(i)]);
    }

    Py_ssize_t size() const { return py::cast<Py_ssize_t>(attr("size")); }

    NumpyDtype dtype() const { return py::borrow<NumpyDtype>(attr("dtype")); }

    bool is_c_contiguous() const
    {
        return py::cast<bool>(attr("flags").attr("c_contiguous"));
    }

    /// Return self if C-contiguous, else a C-ordered copy (keeps 0-d arrays 0-d).
    NumpyArray as_c_contiguous() const
    {
        if (is_c_contiguous())
            return *this;
        return py::steal<NumpyArray>(attr("copy")("C").release());
    }

    /// Pointer to the first element. The array must be C-contiguous.
    void const* data() const { return buffer_ptr(ptr(), PyBUF_SIMPLE); }

    /// Writable pointer to the first element. The array must be C-contiguous and writable.
    void* mutable_data() { return buffer_ptr(ptr(), PyBUF_WRITABLE); }
};

} // namespace tenpy_hdf5
