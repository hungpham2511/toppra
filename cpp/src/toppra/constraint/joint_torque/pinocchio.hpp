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

/** Implementation of JointTorque using pinocchio::rnea function.
 * \extends JointTorque
 * */
template<typename Model = pinocchio::Model>
class Pinocchio;

template<typename _Model>
class Pinocchio : public JointTorque {
  public:
    typedef _Model Model;
    typedef typename Model::Data Data;

    std::ostream& print(std::ostream& os) const
    {
      return JointTorque::print(os << "Pinocchio - ");
    }

    void computeInverseDynamics (const Vector& q, const Vector& v, const Vector& a,
        Vector& tau)
    {
      tau = pinocchio::rnea(m_model, m_data, q, v, a);
    }

    Pinocchio (const Model& model, const Vector& frictionCoeffs = Vector())
      : JointTorque (-model.effortLimit, model.effortLimit, frictionCoeffs)
      , m_model (model)
      , m_data (model)
    {
    }

    /// Move-assignment operator
    Pinocchio (Pinocchio&& other)
      : JointTorque(other)
      , m_storage (std::move(other.m_storage))
      , m_model (other.m_model)
      , m_data (std::move(other.m_data))
    {}

    Pinocchio (const std::string& urdfFilename, const Vector& friction = Vector())
      : Pinocchio (makeModel(urdfFilename), friction) {}

    const Model& model() const { return m_model; }

  private:
    /// Build a pinocchio::Model
    /// \param urdfFilename path to a URDF file.
    static Model* makeModel (const std::string& urdfFilename)
    {
      Model* model (new Model);
      pinocchio::urdf::buildModel(urdfFilename, *model);
      return model;
    }

    /// Constructor that takes ownership of the model
    Pinocchio (Model* model, const Vector& friction)
      : JointTorque (-model->effortLimit, model->effortLimit, friction)
      , m_storage (model)
      , m_model (*model)
      , m_data (m_model)
    {
    }

    /// Store the pinocchio::Model object, in case this object owns it.
    std::unique_ptr<Model> m_storage;
    const Model& m_model;
    Data m_data;
}; // class Pinocchio

} // namespace jointTorque
} // namespace constraint
} // namespace toppra

#endif
