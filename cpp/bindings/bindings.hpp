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

}  // namespace python
}  // namespace toppra

#endif
