#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <toppra/toppra.hpp>
#include <toppra/geometric_path.hpp>

namespace py = pybind11;

PYBIND11_MODULE(toppra_int, m) {
  m.doc() = "toppra C++ bindings (internal)";
  py::class_<toppra::PiecewisePolyPath>(m, "PiecewisePolyPath")
      .def(py::init<>())
      .def(py::init<const toppra::Matrices&, std::vector<toppra::value_type>>())
      .def("eval_single", &toppra::PiecewisePolyPath::eval_single)
      .def("eval", &toppra::PiecewisePolyPath::eval)
      .def("pathInterval", &toppra::PiecewisePolyPath::pathInterval);
}

