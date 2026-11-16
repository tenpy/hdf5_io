message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS " Project hdf5_io, HDF5 import/export with pybind11 bindings ")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "")

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.0.1)
endif()

option(BUILD_TESTING "Build ctest tests" ON)
option(LINTING "Use clang-tidy for linting" OFF)
# Default ON for scikit-build / pip installs; OFF when embedding as a C++ dependency.
if(DEFINED SKBUILD)
  option(HDF5_IO_BUILD_PYTHON "Build pybind11 module and Python-facing library" ON)
else()
  option(HDF5_IO_BUILD_PYTHON "Build pybind11 module and Python-facing library" ON)
endif()

project(
  hdf5_io
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

include(CMakePrintHelpers)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# #####################################################################
# ## HDF5 + HighFive
# #####################################################################
find_package(HDF5 REQUIRED COMPONENTS C)
cmake_print_variables(HDF5_VERSION HDF5_INCLUDE_DIRS)
# object_token uses H5O token APIs on HDF5 >= 1.12 and (fileno, addr) on older releases
# (manylinux_2_28 EPEL still ships 1.10.x).
if(HDF5_VERSION VERSION_LESS "1.10")
  message(FATAL_ERROR "hdf5_io requires HDF5 >= 1.10 (found ${HDF5_VERSION})")
endif()

find_package(HighFive CONFIG QUIET)
if(NOT HighFive_FOUND)
  message(STATUS "HighFive not found; fetching highfive-devs/highfive")
  include(FetchContent)
  set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
    GIT_TAG v3.2.0
  )
  FetchContent_MakeAvailable(HighFive)
endif()

# #####################################################################
# ## hdf5_io_core (Python-free)
# #####################################################################
add_library(hdf5_io_core STATIC)
set_property(TARGET hdf5_io_core PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET hdf5_io_core PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET hdf5_io_core PROPERTY VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(hdf5_io_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
)
target_compile_options(hdf5_io_core PUBLIC -fPIC)

target_include_directories(hdf5_io_core
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_sources(hdf5_io_core PRIVATE
  src/cpp/h5_ops.cpp
  src/cpp/core_io.cpp
)

target_link_libraries(hdf5_io_core PUBLIC HDF5::HDF5)
if(TARGET HighFive::HighFive)
  target_link_libraries(hdf5_io_core PUBLIC HighFive::HighFive)
elseif(TARGET HighFive)
  target_link_libraries(hdf5_io_core PUBLIC HighFive)
else()
  message(FATAL_ERROR "HighFive CMake target not found")
endif()

add_library(hdf5_io::core ALIAS hdf5_io_core)

# #####################################################################
# ## Python / pybind11 (optional)
# #####################################################################
if(HDF5_IO_BUILD_PYTHON)
  set(Python_FIND_VIRTUALENV ONLY)
  set(Python_FIND_STRATEGY LOCATION)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  cmake_print_variables(Python_VERSION)

  add_library(hdf5_io_py STATIC)
  set_property(TARGET hdf5_io_py PROPERTY C_VISIBILITY_PRESET hidden)
  set_property(TARGET hdf5_io_py PROPERTY CXX_VISIBILITY_PRESET hidden)
  set_property(TARGET hdf5_io_py PROPERTY VISIBILITY_INLINES_HIDDEN ON)
  target_compile_options(hdf5_io_py PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
  )
  target_compile_options(hdf5_io_py PUBLIC -fPIC)

  target_include_directories(hdf5_io_py
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

  target_sources(hdf5_io_py PRIVATE
    src/cpp/h5_bridge.cpp
    src/cpp/helpers.cpp
    src/cpp/exportable.cpp
    src/cpp/saver.cpp
    src/cpp/loader.cpp
    src/cpp/wrappers.cpp
  )

  target_link_libraries(hdf5_io_py PUBLIC hdf5_io_core pybind11::pybind11)

  # Back-compat alias for older link lines.
  add_library(hdf5_io ALIAS hdf5_io_py)

  pybind11_add_module(_core pybind/_core.cpp)
  target_include_directories(_core
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind
    PUBLIC ${PROJECT_SOURCE_DIR}/include
  )
  add_subdirectory(pybind)
  target_link_libraries(_core PUBLIC hdf5_io_py)

  if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set_target_properties(_core PROPERTIES
      BUILD_RPATH "$ENV{CONDA_PREFIX}/lib"
      INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
    )
  endif()

  install(TARGETS _core LIBRARY DESTINATION hdf5_io)
endif()

# #####################################################################
# ## Install / export (core)
# #####################################################################
install(TARGETS hdf5_io_core
  EXPORT hdf5_ioTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/hdf5_io
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "constants.h"
    PATTERN "exceptions.h"
    PATTERN "h5_ops.h"
    PATTERN "core_io.h"
)

install(EXPORT hdf5_ioTargets
  FILE hdf5_ioTargets.cmake
  NAMESPACE hdf5_io::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hdf5_io
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hdf5_ioConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hdf5_ioConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hdf5_io
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hdf5_ioConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/hdf5_ioConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hdf5_ioConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hdf5_io
)

# #####################################################################
# ## clang-tidy
# #####################################################################
if(LINTING)
  find_program(CLANG_TIDY_EXE "clang-tidy")
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
  endif()
endif()

# #####################################################################
# ## CTest
# #####################################################################
include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests/ctest)
endif()
