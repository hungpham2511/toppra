import pinocchio
import numpy as np
from toppra.cpp import Interpolation

def torque_constraint(robot, scale=1.):
    from toppra.constraint import JointTorqueConstraint
    def inv_dyn (q, v, a):
        return pinocchio.rnea(robot.model, robot.data, q, v, a)
    return JointTorqueConstraint (inv_dyn,
        np.vstack([-scale*robot.model.effortLimit,scale*robot.model.effortLimit]).T,
        np.zeros(robot.model.nv))

def torque_constraint_cpp(robot, scale=1.):
    from toppra.cpp import jointTorque
    jt = jointTorque.Pinocchio(robot.model, np.zeros(robot.model.nv))
    # TODO apply scale (need bindings of lowerBounds and upperBounds)
    jt.discretizationType = Interpolation
    assert scale == 1.
    return jt

def joint_velocity_constraint(robot, scale=1.):
    from toppra.constraint import JointVelocityConstraint
    return JointVelocityConstraint(
        np.vstack([-scale*robot.model.velocityLimit,scale*robot.model.velocityLimit]).T
        )

def joint_velocity_constraint_cpp(robot, scale=1.):
    from toppra.cpp import LinearJointVelocity
    ljv = LinearJointVelocity(-scale*robot.model.velocityLimit,
            scale*robot.model.velocityLimit)
    ljv.discretizationType = Interpolation
    return ljv

def joint_acceleration_constraint(robot, limit):
    from toppra.constraint import JointAccelerationConstraint
    l = np.array([limit, ] * robot.model.nv)
    return JointAccelerationConstraint(np.vstack([-l,l]).T)

def joint_acceleration_constraint_cpp(robot, limit):
    from toppra.cpp import LinearJointAcceleration
    l = np.array([limit, ] * robot.model.nv)
    lja = LinearJointAcceleration(-l, l)
    lja.discretizationType = Interpolation
    return lja

def generate_random_trajectory(robot, npts, maxd):
    ts = [0.,]
    qs = [pinocchio.randomConfiguration(robot.model),]
    while len(qs) < npts:
        qlast = qs[-1]
        q = pinocchio.randomConfiguration(robot.model)
        d = pinocchio.distance(robot.model, qlast, q)
        if d > maxd:
            q = pinocchio.interpolate(robot.model, qlast, q, maxd / d)
        qs.append(q)
        ts.append(ts[-1] + pinocchio.distance(robot.model, qlast,q))

    # Compute velocities and accelerations
    vs = [np.zeros(robot.model.nv),]
    accs = [np.zeros(robot.model.nv),]
    eps = .01
    for q0, q1, q2 in zip(qs,qs[1:],qs[2:]):
        qprev = pinocchio.interpolate(robot.model, q0, q1,
                1 - eps / pinocchio.distance(robot.model, q0, q1))
        qnext = pinocchio.interpolate(robot.model, q1, q2,
                eps / pinocchio.distance(robot.model, q1, q2))
        # (qnext - qprev) / eps
        vs.append(pinocchio.difference(robot.model, qprev, qnext) / eps)
        # ((qnext - q) - (q - qprev)) / eps^2
        accs.append((pinocchio.difference(robot.model, q, qnext) - pinocchio.difference(robot.model, qprev, q)) / (eps**2))
    vs.append(np.zeros(robot.model.nv))
    accs.append(np.zeros(robot.model.nv))

    from toppra.cpp import PiecewisePolyPath
    return PiecewisePolyPath.constructHermite(qs, vs, ts)
    #from toppra.interpolator import SplineInterpolator as Path
    #return SimplePath(np.array(ts), np.array(qs), np.array(vs), np.array(accs))

if __name__ == "__main__":
    from example_robot_data.robots_loader import loadUR
    robot = loadUR(limited=True)
    path = generate_random_trajectory(robot, 50, 0.3)

    constraints = [
            joint_velocity_constraint(robot),
            joint_acceleration_constraint(robot, 2.),
            torque_constraint(robot),
            ]
    constraints_cpp = [
            joint_velocity_constraint_cpp(robot),
            joint_acceleration_constraint_cpp(robot, 2.),
            torque_constraint_cpp(robot),
            ]

    Ngripoints = 100

    from toppra.algorithm import TOPPRA
    # This does not work yet.
    #algo = TOPPRA(constraints_cpp, path, solver_wrapper='seidel', gridpoints=
    #        np.linspace(0, path.duration, Ngripoints+1))
    algo = TOPPRA(constraints, path, solver_wrapper='seidel', gridpoints=
            np.linspace(0, path.duration, Ngripoints+1))
    sdd, sd, _ = algo.compute_parameterization(0, 0)

    from toppra.cpp import TOPPRA as TOPPRAcpp
    algocpp = TOPPRAcpp(constraints_cpp, path)
    algocpp.setN(Ngripoints)
    retcode = algocpp.computePathParametrization(0, 0)
    sdcpp = np.sqrt(algocpp.parametrizationData.parametrization)

    a = np.vstack([sd,sdcpp,sd-sdcpp]).T
    print(a)
