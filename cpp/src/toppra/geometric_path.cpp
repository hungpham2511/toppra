#include "toppra/geometric_path.hpp"
#include "toppra/toppra.hpp"
#include <list>
#include <iterator>
#include <vector>
namespace toppra {

Vectors GeometricPath::eval(const Vector &positions, int order) const {
  Vectors outputs;
    outputs.resize(positions.size());
    for (size_t i = 0; i < positions.size(); i++) {
      outputs[i] = eval_single(positions(i), order);
    }
    return outputs;
  };


Vector GeometricPath::proposeGridpoints(double max_err_threshold, int max_iteration, double max_seg_length, int min_nb_points, Vector initialGridpoints) const {
  std::list<value_type> gridpoints;
  //gridpoints.reserve(std::max(initialGridpoints.size(), min_nb_points));
  const Bound I = pathInterval();
  if (initialGridpoints.size() == 0) {
    gridpoints.push_front(I[1]);
    gridpoints.push_front(I[0]);
  } else {
    if (initialGridpoints.size() == 1)
      throw std::invalid_argument("initialGridpoints should be empty or have at least 2 elements");
    int N = initialGridpoints.size() - 1;
    for (int i : {0, 1}) {
      if (std::abs(I[i] - initialGridpoints[i*N]) > TOPPRA_NEARLY_ZERO) {
        std::ostringstream oss;
        oss << "initialGridpoints[" << i*N << "] must be " << I[i] << " and not " << initialGridpoints[i*N];
        throw std::invalid_argument(oss.str());
      }
    }
    if ((initialGridpoints.tail(N).array() <= initialGridpoints.head(N).array()).any())
      throw std::invalid_argument("initialGridpoints should be monotonically increasing.");
    for (int i = N; i >= 0; --i)
      gridpoints.push_front(initialGridpoints[i]);
  }

  // Add points according to error threshold
  for (int iter=0; iter < max_iteration; iter++){
    bool add_new_points = false;
    for (auto point = gridpoints.begin(); std::next(point, 1) != gridpoints.end(); point++){

      auto next = std::next(point, 1);

      value_type p_mid = 0.5 * (*point + *next);
      auto dist = (*next - *point);

      if (dist > max_seg_length){
        gridpoints.emplace(next, p_mid);
        add_new_points = true;
        continue;
      }

      // maximum interpolation error
      auto max_err = (0.5 * eval_single(p_mid, 2) * dist * dist).cwiseAbs().maxCoeff();
      if (max_err > max_err_threshold){
        gridpoints.emplace(next, p_mid);
        add_new_points = true;
        continue;
      }
    }

    if (!add_new_points) break;
  }

  // Add points according to smallest number of points
  while (gridpoints.size() < min_nb_points){
    for(auto point=gridpoints.begin(); std::next(point, 1) != gridpoints.end(); std::advance(point, 2)){
      auto next = std::next(point, 1);
      value_type p_mid = 0.5 * (*point + *next);
      gridpoints.emplace(next, p_mid);
    }
  }

  // Return the Eigen vector
  std::vector<value_type> result(gridpoints.begin(), gridpoints.end());
  return Eigen::Map<toppra::Vector, Eigen::Unaligned>(result.data(), result.size());
}
}
