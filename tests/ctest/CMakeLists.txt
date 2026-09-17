add_executable(test_core_roundtrip test_core_roundtrip.cpp)
target_link_libraries(test_core_roundtrip PRIVATE hdf5_io_core)
add_test(NAME hdf5_io_core_roundtrip COMMAND test_core_roundtrip)

if(HDF5_IO_BUILD_PYTHON AND Python_EXECUTABLE)
  add_test(NAME hdf5_io_smoke COMMAND ${Python_EXECUTABLE} -c
    "import hdf5_io; assert hdf5_io.REPR_LIST == 'list'")
endif()
