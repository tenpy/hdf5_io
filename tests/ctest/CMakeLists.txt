if(BUILD_TESTING)
  add_test(NAME hdf5_io_smoke COMMAND ${Python_EXECUTABLE} -c
    "import hdf5_io; assert hdf5_io.REPR_LIST == 'list'")
endif()
