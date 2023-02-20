#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

#include <bindings.hpp>
#include <string>
#include <toppra/geometric_path.hpp>

namespace py = pybind11;

namespace toppra {
namespace python {

nparr path_eval (const GeometricPath& p, const Vector& xs, int order=0)
{
  return toNumpyArray(p.eval(xs,order));
}

template<int order>
nparr path_eval_tpl (const GeometricPath& p, const Vector& xs)
{
  return toNumpyArray(p.eval(xs,order));
}

void exposePaths(py::module m)
{
  py::class_<GeometricPath, GeometricPathPtr >(m, "GeometricPath")
      .def("eval_single", &GeometricPath::eval_single, py::arg("x"), py::arg("order") = 0)
      .def("eval", &path_eval, py::arg("xs"), py::arg("order") = 0)
      .def("evald", &path_eval_tpl<1>)
      .def("evaldd", &path_eval_tpl<2>)
      .def("__call__", &path_eval, py::arg("xs"), py::arg("order") = 0)
      .def("__call__", &GeometricPath::eval_single, py::arg("x"), py::arg("order") = 0)

      .def("serialize", [](const GeometricPath& p) -> py::bytes {
            std::ostringstream ss;
            p.serialize(ss);
            return ss.str();
          })
      .def("deserialize", [](GeometricPath& p, const py::bytes& b) {
            std::stringstream ss;
            ss << b.cast<std::string>();
            p.deserialize(ss);
          })
      .def("proposeGridpoints", &GeometricPath::proposeGridpoints,
          py::arg("maxErrThreshold") = 1e-4,
          py::arg("maxIteration") = 100,
          py::arg("maxSegLength") = 0.05,
          py::arg("minNbPoints") = 100,
          py::arg("initialGridpoints") = Vector()
       )

      .def_property_readonly("dof", &GeometricPath::dof)
      .def_property_readonly("path_interval", &GeometricPath::pathInterval)
      .def_property_readonly("duration",
          [] (const GeometricPath& p) -> double {
            Bound bd (p.pathInterval());
            return bd[1] - bd[0];
          })

      ;

  py::class_<PiecewisePolyPath, std::shared_ptr<PiecewisePolyPath>, GeometricPath>(m, "PiecewisePolyPath")
      .def(py::init<>())
      .def(py::init<const toppra::Matrices&, std::vector<toppra::value_type>>())
      .def("__str__", [](const PiecewisePolyPath& p) -> std::string { return "PiecewisePolyPath(...)"; })
      .def("__repr__", [](const PiecewisePolyPath& p) -> std::string { return "PiecewisePolyPath(...)"; })
      .def_static("constructHermite", &PiecewisePolyPath::constructHermite)
      .def_static("CubicHermiteSpline", &PiecewisePolyPath::CubicHermiteSpline)
      .def_static("CubicSpline", &PiecewisePolyPath::CubicSpline)
      ;
}
}  // namespace python
}  // namespace toppra
