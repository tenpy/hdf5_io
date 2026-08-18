#pragma once

#include <stdexcept>
#include <string>

namespace hdf5_io {

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

} // namespace hdf5_io
