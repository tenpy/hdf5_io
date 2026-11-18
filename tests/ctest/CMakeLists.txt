add_executable(test_core_roundtrip test_core_roundtrip.cpp)
target_link_libraries(test_core_roundtrip PRIVATE tenpy_hdf5_core)
add_test(NAME tenpy_hdf5_core_roundtrip COMMAND test_core_roundtrip)

if(TENPY_HDF5_BUILD_PYTHON AND Python_EXECUTABLE)
  add_test(NAME tenpy_hdf5_smoke COMMAND ${Python_EXECUTABLE} -c
    "import tenpy_hdf5; assert tenpy_hdf5.REPR_LIST == 'list'")
endif()
