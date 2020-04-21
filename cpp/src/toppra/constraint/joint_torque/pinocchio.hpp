#ifndef TOPPRA_CONSTRAINT_JOINT_TORQUE_PINOCCIO_HPP
#define TOPPRA_CONSTRAINT_JOINT_TORQUE_PINOCCIO_HPP

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <toppra/constraint/joint_torque.hpp>

namespace toppra {
namespace constraint {
namespace jointTorque {

template<int _Options=0, template<typename,int> class JointCollectionTpl=pinocchio::JointCollectionDefaultTpl>
class Pinocchio;

/** Implementation of JointTorque using pinocchio::rnea function.
 * */
template<int _Options, template<typename,int> class JointCollectionTpl>
class Pinocchio : public JointTorque {
  public:
    typedef pinocchio::ModelTpl<value_type, _Options, JointCollectionTpl> Model;
    typedef pinocchio::DataTpl <value_type, _Options, JointCollectionTpl> Data;

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
