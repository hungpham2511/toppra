#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

#include <bindings.hpp>
#include <string>
#include <toppra/algorithm.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>

namespace py = pybind11;

namespace toppra {
namespace python {

void exposePaths(py::module m);
void exposeConstraints(py::module m);

PYBIND11_MODULE(toppra_int, m) {
  m.doc() = "toppra C++ bindings (internal)";

  exposePaths(m);
  exposeConstraints(m);

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

  py::class_<PathParametrizationAlgorithm>(m, "PathParametrizationAlgorithm")
      .def("setN", &algorithm::TOPPRA::setN)
      .def("setGridpoints", &PathParametrizationAlgorithm::setGridpoints)
      .def("solver", &PathParametrizationAlgorithm::solver)
      .def("setInitialXBounds", &PathParametrizationAlgorithm::setInitialXBounds)
      .def("computePathParametrization", &algorithm::TOPPRA::computePathParametrization,
           py::arg("vel_start") = 0, py::arg("vel_end") = 0)
      .def_property_readonly("parametrizationData", &algorithm::TOPPRA::getParameterizationData);

  py::class_<algorithm::TOPPRA, PathParametrizationAlgorithm>(m, "TOPPRA")
      .def(py::init<LinearConstraintPtrs, const GeometricPathPtr &>());
}
}  // namespace python
}  // namespace toppra
