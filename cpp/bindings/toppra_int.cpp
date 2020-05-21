#include "toppra/constraint.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <string>
#include <toppra/toppra.hpp>
#include <toppra/geometric_path.hpp>

namespace py = pybind11;

class PyPiecewisePolyPath: public toppra::PiecewisePolyPath {
 public:
  std::string __str__(){
    return "PyPiecewisePolyPath(...)";
  }
  std::string __repr__(){
    return "PyPiecewisePolyPath(...)";
  }
};


PYBIND11_MODULE(toppra_int, m) {
  m.doc() = "toppra C++ bindings (internal)";
  py::class_<toppra::PiecewisePolyPath>(m, "PiecewisePolyPath")
      .def(py::init<>())
      .def(py::init<const toppra::Matrices&, std::vector<toppra::value_type>>())
      .def("eval_single", &toppra::PiecewisePolyPath::eval_single)
      .def("eval", &toppra::PiecewisePolyPath::eval)
      .def("pathInterval", &toppra::PiecewisePolyPath::pathInterval)
      ;

  py::enum_<toppra::DiscretizationType>(m, "DiscretizationType")
      .value("Collocation", toppra::DiscretizationType::Collocation)
      .value("Interpolation", toppra::DiscretizationType::Interpolation).export_values();


  // experimental /////////////////////////////////////////////////////////////
  py::class_<PyPiecewisePolyPath>(m, "PyPiecewisePolyPath")
      .def(py::init<>())
      .def("__str__", &PyPiecewisePolyPath::__str__)
      .def("__repr__", &PyPiecewisePolyPath::__repr__)
      ;
  
}
