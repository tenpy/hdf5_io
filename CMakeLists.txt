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

project(
  hdf5_io
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

include(CMakePrintHelpers)

# #####################################################################
# ## Python / pybind11
# #####################################################################
set(Python_FIND_VIRTUALENV ONLY)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
cmake_print_variables(Python_VERSION)

# #####################################################################
# ## HDF5 + HighFive
# #####################################################################
find_package(HDF5 REQUIRED COMPONENTS C)
cmake_print_variables(HDF5_VERSION HDF5_INCLUDE_DIRS)

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
# ## C++ LIBRARY
# #####################################################################
add_library(hdf5_io STATIC)
set_property(TARGET hdf5_io PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET hdf5_io PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET hdf5_io PROPERTY VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(hdf5_io PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
)
target_compile_options(hdf5_io PUBLIC -fPIC)

target_include_directories(hdf5_io
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_subdirectory(src/cpp)

target_link_libraries(hdf5_io PUBLIC
  pybind11::pybind11
  HDF5::HDF5
)
if(TARGET HighFive::HighFive)
  target_link_libraries(hdf5_io PUBLIC HighFive::HighFive)
elseif(TARGET HighFive)
  target_link_libraries(hdf5_io PUBLIC HighFive)
else()
  message(FATAL_ERROR "HighFive CMake target not found")
endif()

# #####################################################################
# ## PYTHON BINDINGS
# #####################################################################
pybind11_add_module(_core pybind/_core.cpp)
target_include_directories(_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)
add_subdirectory(pybind)
target_link_libraries(_core PUBLIC hdf5_io)

if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  set_target_properties(_core PROPERTIES
    BUILD_RPATH "$ENV{CONDA_PREFIX}/lib"
    INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
  )
endif()

install(TARGETS _core LIBRARY DESTINATION hdf5_io)

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
