#ifndef TOPPRA_BINDINGS_HPP
#define TOPPRA_BINDINGS_HPP

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

#include <bindings.hpp>
#include <string>
#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>

namespace py = pybind11;

namespace toppra {
namespace python {

using nparr = py::array_t<value_type>;

// convert internal types to numpy array
nparr toNumpyArray(const toppra::Vectors& ret);

// wrapper
class PyPiecewisePolyPath {
 protected:
  toppra::PiecewisePolyPath m_path;

 public:
  PyPiecewisePolyPath() = default;
  PyPiecewisePolyPath(const Matrices& coefficients,
                      std::vector<value_type> breakpoints);

  Vector eval_single(value_type x, int order = 0) const;

  nparr eval(const Vector& xs, int order = 0) const;
  Bound pathInterval() const;

  int dof() const;

  std::string __str__();
  std::string __repr__();
};

}  // namespace python
}  // namespace toppra

#endif
