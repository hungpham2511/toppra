#ifndef TOPPRA_TOPPRA_HPP
#define TOPPRA_TOPPRA_HPP

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <toppra/export.hpp>


#ifdef TOPPRA_DEBUG_ON
#define TOPPRA_LOG_DEBUG(X) std::cout << "[DEBUG]: " <<  X << std::endl
#else
#define TOPPRA_LOG_DEBUG(X) ((void)0)
#endif

#if defined(TOPPRA_DEBUG_ON) || defined(TOPPRA_WARN_ON)
#define TOPPRA_LOG_WARN(X) std::cout << "[WARN]: " <<  X << std::endl
#else
#define TOPPRA_LOG_WARN(X) ((void)0)
#endif

#define TOPPRA_UNUSED(x) (void)(x)

// Use for checking if a quantity is very close to zero
#define TOPPRA_NEARLY_ZERO 1e-8
#define TOPPRA_REL_TOL 1e-6
#define TOPPRA_ABS_TOL 1e-8


/// The TOPP-RA namespace
namespace toppra {
  /// The scalar type
  typedef double value_type;

  constexpr value_type infty = std::numeric_limits<value_type>::infinity();

  /// Column vector type
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> Vector;
  /// Matrix type
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;

  /// Vector of Vector
  typedef std::vector<Vector, Eigen::aligned_allocator<Vector> > Vectors;
  /// Vector of Matrix
  typedef std::vector<Matrix, Eigen::aligned_allocator<Matrix> > Matrices;

  // internal data rep for Matrix
  typedef std::tuple<Eigen::Index, Eigen::Index, std::vector<value_type>> MatrixData;
  typedef std::vector<MatrixData> MatricesData;

  /// 2D vector that stores the upper and lower bound of a variable.
  typedef Eigen::Matrix<value_type, 1, 2> Bound;
  /// Vector of Bound
  typedef std::vector<Bound, Eigen::aligned_allocator<Bound> > Bounds;

  class LinearConstraint;
  /// Shared pointer to a LinearConstraint
  typedef std::shared_ptr<LinearConstraint> LinearConstraintPtr;
  /// Vector of LinearConstraintPtr
  typedef std::vector<LinearConstraintPtr> LinearConstraintPtrs;
  namespace constraint {
    class LinearJointVelocity;
    class LinearJointAcceleration;
    class JointTorque;
  } // namespace constraint

  class Solver;
  typedef std::shared_ptr<Solver> SolverPtr;
  namespace solver {
    class qpOASESWrapper;
  } // namespace solver

  class GeometricPath;
  typedef std::shared_ptr<GeometricPath> GeometricPathPtr;

  class PathParametrizationAlgorithm;
  typedef std::shared_ptr<PathParametrizationAlgorithm> PathParametrizationAlgorithmPtr;

} // namespace toppra

#endif
