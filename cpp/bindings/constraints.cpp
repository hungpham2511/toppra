#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

#include <string>
#include <toppra/constraint.hpp>
#include <toppra/constraint/joint_torque.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>

namespace py = pybind11;

#ifdef BUILD_WITH_PINOCCHIO
// Mind that pinocchio headers **must** be included before boost headers.
#include <toppra/constraint/joint_torque/pinocchio.hpp>
#ifdef PINOCCHIO_WITH_PYTHON_INTERFACE
#include <boost/python.hpp>
#endif // PINOCCHIO_WITH_PYTHON_INTERFACE
#endif // BUILD_WITH_PINOCCHIO

namespace toppra {
namespace python {

void exposeConstraints(py::module m)
{
  py::enum_<toppra::DiscretizationType>(m, "DiscretizationType")
      .value("Collocation", toppra::DiscretizationType::Collocation)
      .value("Interpolation", toppra::DiscretizationType::Interpolation)
      .export_values();

  // Abstract class must be binded for derived classes to work
  py::class_<LinearConstraint, LinearConstraintPtr>(m, "LinearConstraint")
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
                        LinearConstraint::discretizationType)
    ;
  py::class_<constraint::LinearJointVelocity,
    std::shared_ptr<constraint::LinearJointVelocity>,
    LinearConstraint>(
      m, "LinearJointVelocity")
      .def(py::init<const Vector &, const Vector &>())
      ;
  py::class_<constraint::LinearJointAcceleration,
    std::shared_ptr<constraint::LinearJointAcceleration>,
    LinearConstraint>(
      m, "LinearJointAcceleration")
      .def(py::init<const Vector &, const Vector &>())
      ;

  py::class_<constraint::JointTorque,
    std::shared_ptr<constraint::JointTorque>,
    LinearConstraint>(
      m, "JointTorque");

  {
    auto mod_jointTorque = m.def_submodule("jointTorque");
#ifdef BUILD_WITH_PINOCCHIO
    py::module::import("pinocchio");
    py::class_<constraint::jointTorque::Pinocchio<>,
      std::shared_ptr<constraint::jointTorque::Pinocchio<> >,
      constraint::JointTorque>(
        mod_jointTorque, "Pinocchio")
#ifdef PINOCCHIO_WITH_PYTHON_INTERFACE
        .def(py::init([](py::object model, const toppra::Vector& fc) {
              return toppra::constraint::jointTorque::Pinocchio<>(
                  boost::python::extract<const pinocchio::Model&>(model.ptr()), fc);
            }))
#endif
        .def(py::init(&constraint::jointTorque::Pinocchio<>::fromURDF))
      ;
#endif
  }
}
}  // namespace python
}  // namespace toppra
