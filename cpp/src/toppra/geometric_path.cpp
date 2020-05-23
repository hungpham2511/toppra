#include "toppra/geometric_path.hpp"
#include "toppra/toppra.hpp"
namespace toppra {

Vectors GeometricPath::eval(const Vector &positions, int order) const {
  Vectors outputs;
    outputs.resize(positions.size());
    for (size_t i = 0; i < positions.size(); i++) {
      outputs[i] = eval_single(positions(i), order);
    }
    return outputs;
  };
}
