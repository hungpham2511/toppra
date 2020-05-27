#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

#include <bindings.hpp>
#include <string>
#include <toppra/constraint.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>

namespace py = pybind11;

namespace toppra {
namespace python {

PYBIND11_MODULE(toppra_int, m) {
  m.doc() = "toppra C++ bindings (internal)";
  py::enum_<toppra::DiscretizationType>(m, "DiscretizationType")
      .value("Collocation", toppra::DiscretizationType::Collocation)
      .value("Interpolation", toppra::DiscretizationType::Interpolation)
      .export_values();

  py::class_<PyPiecewisePolyPath>(m, "PiecewisePolyPath")
      .def(py::init<>())
      .def(py::init<const toppra::Matrices&, std::vector<toppra::value_type>>())
      .def("eval_single", &PyPiecewisePolyPath::eval_single)
      .def("eval", &PyPiecewisePolyPath::eval)
      .def("serialize", &PyPiecewisePolyPath::serialize)
      .def("deserialize", &PyPiecewisePolyPath::deserialize)
      .def("__call__", &PyPiecewisePolyPath::eval, py::arg("xs"), py::arg("order") = 0)
      .def("__str__", &PyPiecewisePolyPath::__str__)
      .def("__repr__", &PyPiecewisePolyPath::__repr__)
      .def_property_readonly("dof", &PyPiecewisePolyPath::dof)
      .def_property_readonly("path_interval", &PyPiecewisePolyPath::pathInterval)
      .def_static("constructHermite", &PyPiecewisePolyPath::constructHermite);

  // Abstract class must be binded for derived classes to work
  py::class_<LinearConstraint>(m, "_LinearConstraint");
  py::class_<constraint::LinearJointVelocity, LinearConstraint>(m,
                                                                "LinearJointVelocity")
      .def(py::init<const Vector&, const Vector&>())
      .def_property_readonly("nbConstraints", &LinearConstraint::nbConstraints)
      .def_property_readonly("nbVariables", &LinearConstraint::nbVariables)
      .def_property_readonly("hasLinearInequalities",
                             &LinearConstraint::hasLinearInequalities)
      .def_property_readonly("hasUbounds", &LinearConstraint::hasUbounds)
      .def_property_readonly("hasXbounds", &LinearConstraint::hasXbounds)
      .def_property("discretizationType",
                    (DiscretizationType(LinearConstraint::*)() const) &
                        LinearConstraint::discretizationType,
                    (void (LinearConstraint::*)(DiscretizationType)) &
                        LinearConstraint::discretizationType);
  py::class_<constraint::LinearJointAcceleration, LinearConstraint>(
      m, "LinearJointAcceleration")
      .def(py::init<const Vector&, const Vector&>())
      .def_property_readonly("nbConstraints", &LinearConstraint::nbConstraints)
      .def_property_readonly("nbVariables", &LinearConstraint::nbVariables)
      .def_property_readonly("hasLinearInequalities",
                             &LinearConstraint::hasLinearInequalities)
      .def_property_readonly("hasUbounds", &LinearConstraint::hasUbounds)
      .def_property_readonly("hasXbounds", &LinearConstraint::hasXbounds)
      .def_property("discretizationType",
                    (DiscretizationType(LinearConstraint::*)() const) &
                        LinearConstraint::discretizationType,
                    (void (LinearConstraint::*)(DiscretizationType)) &
                        LinearConstraint::discretizationType);
}
}  // namespace python
}  // namespace toppra
