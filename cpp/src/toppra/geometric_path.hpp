#ifndef TOPPRA_GEOMETRIC_PATH_HPP
#define TOPPRA_GEOMETRIC_PATH_HPP

#include <cstddef>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <toppra/algorithm.hpp>
#include <toppra/toppra.hpp>
#include <vector>

namespace toppra {


/**
 * \brief Abstract interface for geometric paths.
 */
class GeometricPath {
public:

  GeometricPath()  = default;

  /**
   * Constructor of GeometricPath on vector spaces.
   */
  GeometricPath(int nDof) : m_configSize(nDof), m_dof (nDof) {}

  /**
   * Constructor of GeometricPath on non-vector spaces.
   */
  GeometricPath(int configSize, int nDof) : m_configSize(configSize), m_dof (nDof) {}

  /**
   * \brief Evaluate the path at given position.
   */
  virtual Vector eval_single(value_type, int order = 0) const = 0;

  /**
   * \brief Evaluate the path at given positions (vector).
   *
   * Default implementation: Evaluation each point one-by-one.
   */
  virtual Vectors eval(const Vector &positions, int order = 0) const;

  /**
     \brief Generate gridpoints that sufficiently cover the given path.
     
     This function operates in multiple passes through the geometric
     path from the start to the end point. In each pass, for each
     segment, the maximum interpolation error is estimated using the
     following equation:

        err_{est} = 0.5 * \mathrm{max}(\mathrm{abs}(p'' * d_{segment} ^ 2))

     Here `p''` is the second derivative of the path and d_segment is
     the length of the segment. If the estimated error `err_{test}` is
     greater than the given threshold `max_err_threshold` then the
     segment is divided in two half.
     
     Intuitively, at positions with higher curvature, there must be
     more points in order to improve approximation
     quality. Theoretically toppra performs the best when the proposed
     gridpoint is optimally distributed.

     @param maxErrThreshold Maximum worstcase error thrshold allowable.
     @param maxIteration Maximum number of iterations.
     @param maxSegLength All segments length should be smaller than this value.
     @param minNbPoints Minimum number of points.
     @return The proposed gridpoints.

   */
  Vector proposeGridpoints(double maxErrThreshold=1e-4, int maxIteration=100, double maxSegLength=0.05, int minNbPoints=100) const;

  /**
   * \brief Dimension of the configuration space
   */
  int configSize() const
  {
    return m_configSize;
  }

  /**
   * \return the number of degrees-of-freedom of the path.
   */
  int dof() const
  {
    return m_dof;
  }

  /**
   * \brief Serialize path to stream.
   */
  virtual void serialize(std::ostream &O) const {};

  /**
   * \brief Deserialize stream to construct path.
   */
  virtual void deserialize(std::istream &I){};

  /**
   * \brief Starting and ending path positions.
   */
  virtual Bound pathInterval() const = 0;

  virtual ~GeometricPath() {}

 protected:
  int m_configSize, m_dof;
};

} // namespace toppra

#endif
