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
      .def_property_readonly("path_interval", &PyPiecewisePolyPath::pathInterval);
}
}  // namespace python
}  // namespace toppra
