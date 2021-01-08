#ifndef TOPPRA_CONSTRAINT_CARTESIAN_VELOCITY_NORM_PINOCCIO_HPP
#define TOPPRA_CONSTRAINT_CARTESIAN_VELOCITY_NORM_PINOCCIO_HPP

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <toppra/constraint/cartesian_velocity_norm.hpp>

namespace toppra {
namespace constraint {
namespace cartesianVelocityNorm {

/** Implementation of CartesianVelocityNorm using pinocchio function.
 * \extends CartesianVelocityNorm
 * */
template<typename Model = pinocchio::Model>
class Pinocchio;

template<typename _Model>
class Pinocchio : public CartesianVelocityNorm {
  public:
    typedef _Model Model;
    typedef typename Model::Data Data;

    std::ostream& print(std::ostream& os) const
    {
      return CartesianVelocityNorm::print(os << "Pinocchio - ");
    }

    void computeVelocity (const Vector& q, const Vector& qdot, Vector& v)
    {
      pinocchio::forwardKinematics(m_model, m_data, q, qdot);
      v = pinocchio::getFrameVelocity(m_model, m_data, m_frame_id, m_reference_frame).toVector();
    }

    /// Constructor for constant velocity limit.
    Pinocchio (const Model& model, const Matrix& S, const double& limit,
        pinocchio::FrameIndex frame,
        pinocchio::ReferenceFrame ref_frame = pinocchio::LOCAL_WORLD_ALIGNED)
      : CartesianVelocityNorm (S, limit)
      , m_model (model)
      , m_data (model)
      , m_frame_id (frame)
      , m_reference_frame (ref_frame)
    {
    }

    /// Move-assignment operator
    Pinocchio (Pinocchio&& other)
      : CartesianVelocityNorm(other)
      , m_model (other.m_model)
      , m_data (std::move(other.m_data))
      , m_frame_id (m_frame_id)
      , m_reference_frame (m_reference_frame)
    {}

    const Model& model() const { return m_model; }

  protected:
    /// Constructor for varying velocity limit.
    Pinocchio (const Model& model,
        pinocchio::FrameIndex frame,
        pinocchio::ReferenceFrame ref_frame = pinocchio::LOCAL_WORLD_ALIGNED)
      : CartesianVelocityNorm ()
      , m_model (model)
      , m_data (model)
      , m_frame_id (frame)
      , m_reference_frame (ref_frame)
    {}


  private:
    const Model& m_model;
    Data m_data;
    pinocchio::FrameIndex m_frame_id;
    pinocchio::ReferenceFrame m_reference_frame;
}; // class Pinocchio

} // namespace cartesianVelocityNorm
} // namespace constraint
} // namespace toppra

#endif
