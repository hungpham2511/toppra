#ifndef TOPPRA_BINDINGS_HPP
#define TOPPRA_BINDINGS_HPP

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include "toppra/algorithm/toppra.hpp"

#include <bindings.hpp>
#include <string>
#include <toppra/constraint.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/toppra.hpp>

namespace py = pybind11;

namespace toppra {
namespace python {

using nparr = py::array_t<value_type>;

// convert internal types to numpy array, assume all eigen matrices
// have the same shape.
nparr toNumpyArray(const toppra::Vectors& ret);
nparr toNumpyArray(const toppra::Matrices& ret);

// wrapper
class PyPiecewisePolyPath {
 public:
  toppra::PiecewisePolyPath m_path;

  PyPiecewisePolyPath() = default;
  PyPiecewisePolyPath(const Matrices& coefficients,
                      std::vector<value_type> breakpoints);
  PyPiecewisePolyPath(const PiecewisePolyPath& path);
  Vector eval_single(value_type x, int order = 0) const;

  nparr eval(const Vector& xs, int order = 0) const;
  Bound pathInterval() const;

  int dof() const;
  // Serialize into a bytes stream.
  py::bytes serialize() const;
  // Deserialize  frm a bytes stream.
  void deserialize(const py::bytes&);
  static PyPiecewisePolyPath constructHermite(const Vectors& positions,
                                              const Vectors& velocities,
                                              const std::vector<value_type> times);
  std::string __str__();
  std::string __repr__();
};

/**
   Wrapper for TOPPRA Algorithm.
 */
class PyTOPPRA {
 public:
  PyTOPPRA(LinearConstraintPtrs constraints, PyPiecewisePolyPath& path);
  PyTOPPRA(const py::list& constraints, PyPiecewisePolyPath& path);
  ReturnCode computePathParametrization(value_type vel_start = 0,
                                        value_type vel_end = 0);
  ParametrizationData getParameterizationData() const;
  void setN(int N);

 private:
  std::unique_ptr<toppra::algorithm::TOPPRA> m_problem;
  LinearConstraintPtr m_tmp;
};

}  // namespace python
}  // namespace toppra

#endif
