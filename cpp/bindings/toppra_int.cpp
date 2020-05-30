#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

#include <bindings.hpp>
#include <string>
#include <toppra/algorithm.hpp>
#include <toppra/constraint.hpp>
#include <toppra/constraint/joint_torque.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>

namespace py = pybind11;

#ifdef BUILD_WITH_PINOCCHIO
// Mind that pinocchio headers **must** be included before boost headers.
#include <toppra/constraint/joint_torque/pinocchio.hpp>
#include <boost/python.hpp>

toppra::constraint::jointTorque::Pinocchio<> createJointTorque(py::object model,
    const toppra::Vector& fc)
{
  return toppra::constraint::jointTorque::Pinocchio<>(
      boost::python::extract<const pinocchio::Model&>(model.ptr()), fc);
}
#endif

namespace toppra {
namespace python {

nparr path_eval (const PiecewisePolyPath& p, const Vector& xs, int order=0)
{
  return toNumpyArray(p.eval(xs,order));
}

template<int order>
nparr path_eval_tpl (const PiecewisePolyPath& p, const Vector& xs)
{
  return toNumpyArray(p.eval(xs,order));
}

PYBIND11_MODULE(toppra_int, m) {
  m.doc() = "toppra C++ bindings (internal)";
  py::enum_<toppra::DiscretizationType>(m, "DiscretizationType")
      .value("Collocation", toppra::DiscretizationType::Collocation)
      .value("Interpolation", toppra::DiscretizationType::Interpolation)
      .export_values();

  py::class_<PiecewisePolyPath, std::shared_ptr<PiecewisePolyPath> >(m, "PiecewisePolyPath")
      .def(py::init<>())
      .def(py::init<const toppra::Matrices&, std::vector<toppra::value_type>>())
      .def("eval_single", &PiecewisePolyPath::eval_single)
      .def("eval", &path_eval, py::arg("xs"), py::arg("order") = 0)
      .def("evald", &path_eval_tpl<1>)
      .def("evald", &path_eval_tpl<2>)
      .def("serialize", [](const PiecewisePolyPath& p) -> py::bytes {
            std::ostringstream ss;
            p.serialize(ss);
            return ss.str();
          })
      .def("deserialize", [](PiecewisePolyPath& p, const py::bytes& b) {
            std::stringstream ss;
            ss << b.cast<std::string>();
            p.deserialize(ss);
          })
      .def("__call__", &path_eval, py::arg("xs"), py::arg("order") = 0)
      .def("__call__", &PiecewisePolyPath::eval_single, py::arg("x"), py::arg("order") = 0)
      .def("__str__", [](const PiecewisePolyPath& p) -> std::string { return "PiecewisePolyPath(...)"; })
      .def("__repr__", [](const PiecewisePolyPath& p) -> std::string { return "PiecewisePolyPath(...)"; })
      .def_property_readonly("dof", &PiecewisePolyPath::dof)
      .def_property_readonly("path_interval", &PiecewisePolyPath::pathInterval)
      .def_property_readonly("duration",
          [] (const PiecewisePolyPath& p) -> double {
            Bound bd (p.pathInterval());
            return bd[1] - bd[0];
          })
      .def_static("constructHermite", &PiecewisePolyPath::constructHermite);

  // Abstract class must be binded for derived classes to work
  py::class_<LinearConstraint, LinearConstraintPtr>(m, "_LinearConstraint")
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
        .def(py::init(&createJointTorque))
      ;
#endif
  }

  // algorithm
  py::enum_<toppra::ReturnCode>(m, "ReturnCode")
      .value("OK", toppra::ReturnCode::OK)
      .value("ERR_UNKNOWN", toppra::ReturnCode::ERR_UNKNOWN)
      .value("ERR_UNINITIALIZED", toppra::ReturnCode::ERR_UNINITIALIZED)
      .value("ERR_FAIL_FEASIBLE", toppra::ReturnCode::ERR_FAIL_FEASIBLE)
      .value("ERR_FAIL_CONTROLLABLE", toppra::ReturnCode::ERR_FAIL_CONTROLLABLE)
      .value("ERR_FAIL_FORWARD_PASS", toppra::ReturnCode::ERR_FAIL_FORWARD_PASS)
      .export_values();

  py::class_<toppra::ParametrizationData>(m, "ParametrizationData")
      .def(py::init<>())
      .def_readwrite("gridpoints", &toppra::ParametrizationData::gridpoints)
      .def_readwrite("parametrization", &toppra::ParametrizationData::parametrization)
      .def_readwrite("controllable_sets",
                     &toppra::ParametrizationData::controllable_sets)
      .def_readwrite("feasible_sets", &toppra::ParametrizationData::feasible_sets)
      .def_readwrite("ret_code", &toppra::ParametrizationData::ret_code);

  py::class_<algorithm::TOPPRA>(m, "TOPPRA")
      .def(py::init<LinearConstraintPtrs, const std::shared_ptr<PiecewisePolyPath> &>())
      .def("computePathParametrization", &algorithm::TOPPRA::computePathParametrization,
           py::arg("vel_start") = 0, py::arg("vel_end") = 0)
      .def("setN", &algorithm::TOPPRA::setN)
      .def_property_readonly("parametrizationData", &algorithm::TOPPRA::getParameterizationData);
}
}  // namespace python
}  // namespace toppra
