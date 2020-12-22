#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

#include <string>
#include <toppra/geometric_path.hpp>
#include <toppra/constraint.hpp>
#include <toppra/constraint/cartesian_velocity_norm.hpp>
#include <toppra/constraint/joint_torque.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>

namespace py = pybind11;

#ifdef BUILD_WITH_PINOCCHIO
// Mind that pinocchio headers **must** be included before boost headers.
#include <toppra/constraint/cartesian_velocity_norm/pinocchio.hpp>
#include <toppra/constraint/joint_torque/pinocchio.hpp>
#ifdef PINOCCHIO_WITH_PYTHON_INTERFACE
#include <boost/python.hpp>
#endif // PINOCCHIO_WITH_PYTHON_INTERFACE
#endif // BUILD_WITH_PINOCCHIO

namespace toppra {
namespace python {

using namespace constraint;

void exposeConstraints(py::module m)
{
  py::enum_<DiscretizationType>(m, "DiscretizationType")
      .value("Collocation", DiscretizationType::Collocation)
      .value("Interpolation", DiscretizationType::Interpolation)
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

      .def ("computeParams", [](LinearConstraint& constraint, const GeometricPath& path, const Vector& gridpoints) -> py::tuple {
            Vectors a, b, c, g;
            Matrices F;
            Bounds ub, xb;
            constraint.computeParams(path, gridpoints, a, b, c, F, g, ub, xb);
            py::list list;
            py::none none;
            if (constraint.hasLinearInequalities()) {
              list.append(a);
              list.append(b);
              list.append(c);
              list.append(F);
              list.append(g);
            } else
              for (int i = 0; i < 5; ++i) list.append(none);
            if (constraint.hasUbounds()) list.append(ub);
            else list.append(none);
            if (constraint.hasXbounds()) list.append(xb);
            else list.append(none);
            return list;
          }
        );
    ;
  py::class_<LinearJointVelocity,
    std::shared_ptr<LinearJointVelocity>,
    LinearConstraint>(
      m, "LinearJointVelocity")
      .def(py::init<const Vector &, const Vector &>())
      ;
  py::class_<LinearJointAcceleration,
    std::shared_ptr<LinearJointAcceleration>,
    LinearConstraint>(
      m, "LinearJointAcceleration")
      .def(py::init<const Vector &, const Vector &>())
      ;

  py::class_<JointTorque,
    std::shared_ptr<JointTorque>,
    LinearConstraint>(
      m, "JointTorque");

  {
    auto mod_jointTorque = m.def_submodule("jointTorque");
#ifdef BUILD_WITH_PINOCCHIO
    using jointTorque::Pinocchio;
#ifdef PINOCCHIO_WITH_PYTHON_INTERFACE
    py::module::import("pinocchio");
#endif
    py::class_<Pinocchio<>, std::shared_ptr<Pinocchio<> >, JointTorque>(
        mod_jointTorque, "Pinocchio")
        .def(py::init<const std::string&, const Vector&>(),
            py::arg("urdfFilename"), py::arg("frictionCoeffs") = Vector())
#ifdef PINOCCHIO_WITH_PYTHON_INTERFACE
        .def(py::init([](py::object model, const Vector& fc) -> Pinocchio<>* {
              return new Pinocchio<>(
                  boost::python::extract<const pinocchio::Model&>(model.ptr()), fc);
            }), py::keep_alive<1, 2>(), py::arg("model"), py::arg("frictionCoeffs") = Vector())
#endif
      ;
#endif
  }

  py::class_<CartesianVelocityNorm,
    std::shared_ptr<CartesianVelocityNorm>,
    LinearConstraint>(
      m, "CartesianVelocityNorm");

  {
    auto mod_jointTorque = m.def_submodule("cartesianVelocityNorm");
#ifdef BUILD_WITH_PINOCCHIO
    using cartesianVelocityNorm::Pinocchio;
#ifdef PINOCCHIO_WITH_PYTHON_INTERFACE
    py::module::import("pinocchio");
#endif
    py::class_<Pinocchio<>, std::shared_ptr<Pinocchio<> >, CartesianVelocityNorm>(
        mod_jointTorque, "Pinocchio")
#ifdef PINOCCHIO_WITH_PYTHON_INTERFACE
        .def(py::init([](py::object model, const Matrix& S, const double& limit, int frame_id) -> Pinocchio<>* {
              return new Pinocchio<>(
                  boost::python::extract<const pinocchio::Model&>(model.ptr()), S, limit, frame_id);
            }), py::keep_alive<1, 2>(), py::arg("model"), py::arg("S"), py::arg("limit"), py::arg("frame_id"))
#endif
      ;
#endif
  }
}
}  // namespace python
}  // namespace toppra
