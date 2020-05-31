#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <bindings.hpp>
#include <cstddef>
#include <memory>
#include <sstream>
#include "toppra/algorithm/toppra.hpp"
#include "toppra/geometric_path/piecewise_poly_path.hpp"
#include "toppra/toppra.hpp"

namespace toppra {
namespace python {

nparr toNumpyArray(const toppra::Vectors& ret) {
  nparr x;
  x.resize({(size_t)ret.size(), (size_t)ret[0].size()});
  for (size_t i = 0; i < x.shape()[0]; i++)
    for (size_t j = 0; j < x.shape()[1]; j++) x.mutable_at(i, j) = ret[i](j);
  return x;
}

nparr toNumpyArray(const toppra::Matrices& ret) {
  nparr x;
  x.resize({(size_t)ret.size(), (size_t)ret[0].rows(), (size_t)ret[0].cols()});
  for (size_t i = 0; i < x.shape()[0]; i++)
    for (size_t j = 0; j < x.shape()[1]; j++)
      for (size_t k = 0; k < x.shape()[2]; k++) x.mutable_at(i, j, k) = ret[i](j, k);
  return x;
}

}  // namespace python
}  // namespace toppra
