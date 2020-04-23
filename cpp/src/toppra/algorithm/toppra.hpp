#ifndef TOPPRA_ALGORITHM_TOPPRA_HPP
#define TOPPRA_ALGORITHM_TOPPRA_HPP

#include <toppra/algorithm.hpp>
#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>
#include "toppra/toppra.hpp"

namespace toppra {
namespace algorithm {
class TOPPRA : public PathParametrizationAlgorithm {
 public:
  TOPPRA(const LinearConstraintPtrs &constraints, const GeometricPath &path);
};
}  // namespace algorithm
}  // namespace toppra

#endif
