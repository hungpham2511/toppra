"""Retime a path subject to cartesian speed constraints
=======================================================

In this example, we will see how can we retime a generic spline-based
path subject to speed constraints on the robot's TCP link.
PyKDL is used for the forward kinematics.

This example is very similar to the plot_kinematics.py example but the 
joint velocity limits are substituted for cartesian velocity limits.

"""

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time
import PyKDL as kdl
import math

ta.setup_logging("INFO")

################################################################################
# Build a robot chain.

# Create a chain for the UR10 DH parameters as an example
chain = kdl.Chain()
chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotZ), kdl.Frame().DH(0,       math.pi/2,  0.1273,   0)))
chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotX), kdl.Frame().DH(-0.612,  0,          0,        0)))
chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotZ), kdl.Frame().DH(-0.5723, 0,          0,        0)))
chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotX), kdl.Frame().DH(0,       math.pi/2,  0.163941, 0)))
chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotZ), kdl.Frame().DH(0,       -math.pi/2, 0.1157,   0)))
chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotX), kdl.Frame().DH(0,       0,          0.0922,   0)))

# Extract the number of joints
dof = chain.getNrOfJoints()

# Create a forward kinematic velocity solver
fk_solver_vel = kdl.ChainFkSolverVel_recursive(chain)

################################################################################
# We generate a path with some random waypoints.

def generate_new_problem(seed=9):
    # Parameters
    N_samples = 5
    np.random.seed(seed)
    way_pts = np.random.randn(N_samples, dof)
    return (
        np.linspace(0, 1, 5),
        way_pts,
        2.0, 4.0,
        10 + np.random.rand(dof) * 2,
    )
ss, way_pts, lin_spd, ang_spd, alims = generate_new_problem()

################################################################################
# Define the callback which calculates TCP velocity from joint positions and
# velocities.

def fk_vel(q, dq):
    # Convert numpy arrays to KDL-compatible JntArrayVel type
    jq = kdl.JntArrayVel(dof)
    for i in range(dof):
        jq.q[i] = q[i]
        jq.qdot[i] = dq[i]
    
    # Object to store the resulting frame & twist
    framevel = kdl.FrameVel()

    # Do the FK
    result = fk_solver_vel.JntToCart(jq, framevel)
    if 0 != result:
        raise Exception(f"Error solving TCP velocity: Error code = {result}")

    # Get the twist and normalize to find absolute speeds
    twist = framevel.GetTwist()
    return twist.vel.Norm(), twist.rot.Norm()

################################################################################
# Define the geometric path and two constraints.

path = ta.SplineInterpolator(ss, way_pts)
pc_cart_vel = constraint.CartesianSpeedConstraint(fk_vel, lin_spd, ang_spd, dof)
pc_acc = constraint.JointAccelerationConstraint(alims)

################################################################################
# We solve the parametrization problem using the
# `ParametrizeConstAccel` parametrizer. This parametrizer is the
# classical solution, guarantee constraint and boundary conditions
# satisfaction.
instance = algo.TOPPRA([pc_cart_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
jnt_traj = instance.compute_trajectory()

################################################################################
# The output trajectory is an instance of
# :class:`toppra.interpolator.AbstractGeometricPath`.
ts_sample = np.linspace(0, jnt_traj.duration, 100)
qs_sample = jnt_traj(ts_sample)
qds_sample = jnt_traj(ts_sample, 1)

# Extract TCP speeds in order to plot
fkv = np.vectorize(fk_vel, signature='(n),(n)->(),()')
linear_spd, angular_spd = fkv(qs_sample, qds_sample)

qdds_sample = jnt_traj(ts_sample, 2)
fig, axs = plt.subplots(4, 1, sharex=True)
for i in range(path.dof):
    # plot the i-th joint trajectory
    axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
    axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
    axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))

# Plot the cartesian linear speed and angular speed of the TCP
axs[3].plot(ts_sample, linear_spd)
axs[3].plot(ts_sample, angular_spd)

axs[0].set_ylabel("Position (rad)")
axs[1].set_ylabel("Velocity (rad/s)")
axs[2].set_ylabel("Acceleration (rad/s2)")
axs[3].set_ylabel("Cartesian velocity (m/s)")
axs[3].set_xlabel("Time (s)")
plt.show()


################################################################################
# Optionally, we can inspect the output.
instance.compute_feasible_sets()
instance.inspect()
