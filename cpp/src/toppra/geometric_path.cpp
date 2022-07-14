#include "toppra/geometric_path.hpp"
#include "toppra/toppra.hpp"
#include <forward_list>
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


Vector GeometricPath::proposeGridpoints(double max_err_threshold, int max_iteration, double max_seg_length, int min_nb_points) const {
  std::forward_list<value_type> gridpoints {pathInterval()[0], pathInterval()[1]};
  int nb_gridpoints = 2;

  // Add points according to error threshold
  for (auto iter=0; iter < max_iteration; iter++){
    bool add_new_points = false;
    for (auto point = gridpoints.begin(); std::next(point, 1) != gridpoints.end(); point++){

      value_type p_mid = 0.5 * (*point + *std::next(point, 1));
      auto dist = (*std::next(point, 1) - *point);

      if (dist > max_seg_length){
        gridpoints.emplace_after(point, p_mid);
        add_new_points = true;
        nb_gridpoints ++;
        continue;
      }

      auto max_err = (0.5 * eval_single(p_mid, 2) * dist * dist).cwiseAbs().maxCoeff();
      if (max_err > max_err_threshold){
        gridpoints.emplace_after(point, p_mid);
        add_new_points = true;
        nb_gridpoints ++;
        continue;
      }
    }

    if (!add_new_points) break;
  }

  // Add points according to smallest number of points
  while (nb_gridpoints < min_nb_points){
    for(auto point=gridpoints.begin(); std::next(point, 1) != gridpoints.end(); std::advance(point, 2)){
      value_type p_mid = 0.5 * (*point + *std::next(point, 1));
      gridpoints.emplace_after(point, p_mid);
      nb_gridpoints++;
    }
  }

  // Return the Eigen vector
  std::vector<value_type> result(gridpoints.begin(), gridpoints.end());
  return Eigen::Map<toppra::Vector, Eigen::Unaligned>(result.data(), result.size());
}
}