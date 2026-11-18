#pragma once

#include <stdexcept>
#include <string>

namespace tenpy_hdf5 {

class Hdf5FormatError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

class Hdf5ExportError : public Hdf5FormatError
{
  public:
    using Hdf5FormatError::Hdf5FormatError;
};

class Hdf5ImportError : public Hdf5FormatError
{
  public:
    using Hdf5FormatError::Hdf5FormatError;
};

} // namespace tenpy_hdf5
