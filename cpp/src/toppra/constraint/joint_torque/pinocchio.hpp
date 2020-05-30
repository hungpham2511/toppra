#ifndef TOPPRA_CONSTRAINT_JOINT_TORQUE_PINOCCIO_HPP
#define TOPPRA_CONSTRAINT_JOINT_TORQUE_PINOCCIO_HPP

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <toppra/constraint/joint_torque.hpp>

namespace toppra {
namespace constraint {
namespace jointTorque {

template<typename Model = pinocchio::Model>
class Pinocchio;

/** Implementation of JointTorque using pinocchio::rnea function.
 * */
template<typename _Model>
class Pinocchio : public JointTorque {
  public:
    typedef _Model Model;
    typedef typename _Model::Data Data;

    /// Build a JointTorque constraint with no friction
    /// \param urdfFilename path to a URDF file.
    static Pinocchio fromURDF (const std::string& urdfFilename)
    {
      Model model;
      pinocchio::urdf::buildModel(urdfFilename, model);
      return Pinocchio(model, Vector::Zero(model.nv));
    }

    std::ostream& print(std::ostream& os) const
    {
      return JointTorque::print(os << "Pinocchio - ");
    }

    void computeInverseDynamics (const Vector& q, const Vector& v, const Vector& a,
        Vector& tau)
    {
      tau = pinocchio::rnea(m_model, m_data, q, v, a);
    }

    Pinocchio (const Model& model, const Vector& frictionCoeffs)
      : JointTorque (-model.effortLimit, model.effortLimit, frictionCoeffs)
      , m_model (model)
      , m_data (model)
    {
    }

  private:
    const Model& m_model;
    Data m_data;
}; // class Pinocchio

} // namespace jointTorque
} // namespace constraint
} // namespace toppra

#endif
