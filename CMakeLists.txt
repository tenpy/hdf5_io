message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS " Project tenpy_hdf5, HDF5 import/export with nanobind bindings ")
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
  option(TENPY_HDF5_BUILD_PYTHON "Build nanobind module and Python-facing library" ON)
else()
  option(TENPY_HDF5_BUILD_PYTHON "Build nanobind module and Python-facing library" ON)
endif()

project(
  tenpy_hdf5
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
  message(FATAL_ERROR "tenpy_hdf5 requires HDF5 >= 1.10 (found ${HDF5_VERSION})")
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
# ## tenpy_hdf5_core (Python-free)
# #####################################################################
add_library(tenpy_hdf5_core STATIC)
set_property(TARGET tenpy_hdf5_core PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET tenpy_hdf5_core PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET tenpy_hdf5_core PROPERTY VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(tenpy_hdf5_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
)
target_compile_options(tenpy_hdf5_core PUBLIC -fPIC)

target_include_directories(tenpy_hdf5_core
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_sources(tenpy_hdf5_core PRIVATE
  src/cpp/h5_ops.cpp
  src/cpp/core_io.cpp
)

target_link_libraries(tenpy_hdf5_core PUBLIC HDF5::HDF5)
if(TARGET HighFive::HighFive)
  target_link_libraries(tenpy_hdf5_core PUBLIC HighFive::HighFive)
elseif(TARGET HighFive)
  target_link_libraries(tenpy_hdf5_core PUBLIC HighFive)
else()
  message(FATAL_ERROR "HighFive CMake target not found")
endif()

add_library(tenpy_hdf5::core ALIAS tenpy_hdf5_core)

# #####################################################################
# ## Python / nanobind (optional)
# #####################################################################
if(TENPY_HDF5_BUILD_PYTHON)
  set(Python_FIND_VIRTUALENV ONLY)
  set(Python_FIND_STRATEGY LOCATION)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
  if(NOT nanobind_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE nanobind_ROOT
      ERROR_QUIET
    )
  endif()
  find_package(nanobind CONFIG REQUIRED)
  cmake_print_variables(Python_VERSION)

  # Stable ABI (abi3, Python >= 3.12). Mirrors the conditions under which nanobind honors
  # STABLE_ABI; tenpy_hdf5_py has to be compiled with the same Py_LIMITED_API as the module.
  set(TENPY_HDF5_STABLE_ABI OFF)
  if(TARGET Python::SABIModule
     AND Python_VERSION VERSION_GREATER_EQUAL "3.12"
     AND NOT NB_ABI MATCHES "[0-9]t")
    set(TENPY_HDF5_STABLE_ABI ON)
  endif()

  add_library(tenpy_hdf5_py STATIC)
  set_property(TARGET tenpy_hdf5_py PROPERTY C_VISIBILITY_PRESET hidden)
  set_property(TARGET tenpy_hdf5_py PROPERTY CXX_VISIBILITY_PRESET hidden)
  set_property(TARGET tenpy_hdf5_py PROPERTY VISIBILITY_INLINES_HIDDEN ON)
  target_compile_options(tenpy_hdf5_py PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
  )
  target_compile_options(tenpy_hdf5_py PUBLIC -fPIC)

  target_include_directories(tenpy_hdf5_py
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

  target_sources(tenpy_hdf5_py PRIVATE
    src/cpp/h5_bridge.cpp
    src/cpp/helpers.cpp
    src/cpp/exportable.cpp
    src/cpp/saver.cpp
    src/cpp/loader.cpp
    src/cpp/wrappers.cpp
  )

  # nanobind headers (header-only part) + Python headers; the nanobind library itself is
  # linked into _core by nanobind_add_module().
  target_include_directories(tenpy_hdf5_py SYSTEM PUBLIC ${NB_DIR}/include)
  target_link_libraries(tenpy_hdf5_py PUBLIC tenpy_hdf5_core Python::Module)
  target_compile_features(tenpy_hdf5_py PUBLIC cxx_std_17)
  if(TENPY_HDF5_STABLE_ABI)
    target_compile_definitions(tenpy_hdf5_py PUBLIC Py_LIMITED_API=0x030C0000)
  endif()

  # Alias for the Python-facing static library.
  add_library(tenpy_hdf5 ALIAS tenpy_hdf5_py)

  if(TENPY_HDF5_STABLE_ABI)
    nanobind_add_module(_core STABLE_ABI pybind/_core.cpp)
  else()
    nanobind_add_module(_core pybind/_core.cpp)
  endif()
  target_include_directories(_core
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind
    PUBLIC ${PROJECT_SOURCE_DIR}/include
  )
  add_subdirectory(pybind)
  target_link_libraries(_core PRIVATE tenpy_hdf5_py)

  if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set_target_properties(_core PROPERTIES
      BUILD_RPATH "$ENV{CONDA_PREFIX}/lib"
      INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
    )
  endif()

  install(TARGETS _core LIBRARY DESTINATION tenpy_hdf5)
endif()

# #####################################################################
# ## Install / export (core)
# #####################################################################
install(TARGETS tenpy_hdf5_core
  EXPORT tenpy_hdf5Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/tenpy_hdf5
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "constants.h"
    PATTERN "exceptions.h"
    PATTERN "h5_ops.h"
    PATTERN "core_io.h"
)

install(EXPORT tenpy_hdf5Targets
  FILE tenpy_hdf5Targets.cmake
  NAMESPACE tenpy_hdf5::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tenpy_hdf5
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tenpy_hdf5Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/tenpy_hdf5Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tenpy_hdf5
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/tenpy_hdf5ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/tenpy_hdf5Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/tenpy_hdf5ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tenpy_hdf5
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
