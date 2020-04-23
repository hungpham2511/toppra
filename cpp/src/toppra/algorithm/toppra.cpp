#include <memory>
#include <toppra/algorithm.hpp>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/toppra.hpp>

namespace toppra {
namespace algorithm {

TOPPRA::TOPPRA(const LinearConstraintPtrs &constraints, const GeometricPath &path)
    : PathParametrizationAlgorithm{constraints, path} {};

}  // namespace algorithm
}  // namespace toppra
