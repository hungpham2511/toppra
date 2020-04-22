#ifndef TOPPRA_TOPPRA_HPP
#define TOPPRA_TOPPRA_HPP

#include <memory>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>

/// The TOPP-RA namespace
namespace toppra {
  /// The scalar type
  typedef double value_type;

  /// Column vector type
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> Vector;
  /// Matrix type
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;

  /// Vector of Vector
  typedef std::vector<Vector, Eigen::aligned_allocator<Vector> > Vectors;
  /// Vector of Matrix
  typedef std::vector<Matrix, Eigen::aligned_allocator<Matrix> > Matrices;

  /// 2D vector that stores the upper and lower bound of a variable.
  typedef Eigen::Matrix<value_type, 1, 2> Bound;
  /// Vector of Bound
  typedef std::vector<Bound, Eigen::aligned_allocator<Bound> > Bounds;

  class LinearConstraint;
  typedef std::shared_ptr<LinearConstraint> LinearConstraintPtr;
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
} // namespace toppra

#endif
