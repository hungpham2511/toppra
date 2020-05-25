#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <bindings.hpp>
#include <cstddef>
#include <sstream>
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
      for (size_t k = 0; j < x.shape()[2]; j++) x.mutable_at(i, j, k) = ret[i](j, k);
  return x;
}

PyPiecewisePolyPath::PyPiecewisePolyPath(const Matrices& coefficients,
                                         std::vector<value_type> breakpoints)
    : m_path{coefficients, breakpoints} {};

PyPiecewisePolyPath::PyPiecewisePolyPath(const PiecewisePolyPath& path)
    : m_path(path){};

Vector PyPiecewisePolyPath::eval_single(value_type x, int order) const {
  return m_path.eval_single(x, order);
};

nparr PyPiecewisePolyPath::eval(const Vector& xs, int order) const {
  auto ret = m_path.eval(xs, order);
  return toNumpyArray(ret);
}

Bound PyPiecewisePolyPath::pathInterval() const { return m_path.pathInterval(); }

int PyPiecewisePolyPath::dof() const { return m_path.dof(); }

py::bytes PyPiecewisePolyPath::serialize() const {
  std::ostringstream ss;
  m_path.serialize(ss);
  return ss.str();
};

void PyPiecewisePolyPath::deserialize(const py::bytes& b) {
  std::stringstream ss;
  ss << b.cast<std::string>();
  m_path.deserialize(ss);
}
PyPiecewisePolyPath PyPiecewisePolyPath::constructHermite(
    const Vectors& positions, const Vectors& velocities,
    const std::vector<value_type> times) {
  auto path_c = PiecewisePolyPath::constructHermite(positions, velocities, times);
  return PyPiecewisePolyPath{path_c};
};
std::string PyPiecewisePolyPath::__str__() { return "PiecewisePolyPath(...)"; }
std::string PyPiecewisePolyPath::__repr__() { return "PiecewisePolyPath(...)"; }

}  // namespace python
}  // namespace toppra
