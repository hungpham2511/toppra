#ifndef TOPPRA_TOPPRA_HPP
#define TOPPRA_TOPPRA_HPP

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace toppra {
  typedef double value_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> Vector;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;

  /// Vector of Vector
  typedef std::vector<Vector, Eigen::aligned_allocator<Vector> > Vectors;
  /// Vector of Matrix
  typedef std::vector<Matrix, Eigen::aligned_allocator<Matrix> > Matrices;

  /// 2D vector that stores the upper and lower bound of a variable.
  typedef Eigen::Matrix<value_type, 1, 2> Bound;
  /// Vector of Bound
  typedef std::vector<Bound, Eigen::aligned_allocator<Bound> > Bounds;
} // namespace toppra

#endif
