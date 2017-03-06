#! /usr/bin/python

import openravepy as orpy
import time
import random


def simple_ik(manip, pose):
    return manip.FindIKSolution(pose, orpy.IkFilterOptions.CheckEnvCollisions)


# def complex_ik_rec(manip, pose, min_v, max_v, depth, free_joint):
#     if depth < 0:
#         return None
#     v = (min_v + max_v) / 2.0
#     robot = manip.GetRobot()
#     robot.SetDOFValues([v], [free_joint])
#     sol = manip.FindIKSolution(pose, orpy.IkFilterOptions.CheckEnvCollisions)
#     if sol is not None:
#         return sol
#     left_sol = complex_ik_rec(manip, pose, min_v, v, depth - 1, free_joint)
#     if left_sol is not None:
#         return left_sol
#     right_sol = complex_ik_rec(manip, pose, v, max_v, depth - 1, free_joint)
#     return right_sol

def complex_ik(manip, pose, max_iterations=2, free_joint_index=0):
    sol = None
    lower_limits, upper_limits = manip.GetRobot().GetDOFLimits()
    min_v, max_v = lower_limits[free_joint_index], upper_limits[free_joint_index]
    stride = (max_v - min_v) / 2.0
    num_steps = 2
    for i in range(max_iterations):
        for j in range(1, num_steps, 2):
            v = min_v + j * stride
            robot.SetDOFValues([v], [free_joint_index])
            sol = manip.FindIKSolution(pose, orpy.IkFilterOptions.CheckEnvCollisions)
            if sol is not None:
                return sol
        num_steps *= 2
        stride /= 2.0
    return sol


def eval_ik_method(manip, robot, test_configurations, ik_method):
    num_success = 0
    start_time = time.time()
    for i in range(len(test_configurations)):
        target_pose = manip.GetEndEffectorTransform()
        robot.SetDOFValues(test_configurations[i], range(7))
        sol = ik_method(manip, target_pose)
        num_success += sol is not None
    total_time = time.time() - start_time
    avg_runtime = total_time / float(len(test_configurations))
    return num_success, avg_runtime


if __name__ == '__main__':
    # TODO read environment etc from configuration file
    # TODO make ROS node
    env = orpy.Environment()
    # env.SetViewer('qtcoin')
    env.Load('../data/environments/test_env.xml')
    robot = env.GetRobots()[0]
    manip = robot.GetManipulator('arm')
    arm_ik = orpy.databases.inversekinematics.InverseKinematicsModel(robot,
                                                                     iktype=orpy.IkParameterization.Type.Transform6D)
    arm_ik.load()
    robot.SetController(None)
    test_configurations = []
    lower_limits, upper_limits = robot.GetDOFLimits()
    num_tests = 1000
    for i in range(num_tests + 1):
        test_config = 7 * [0.0]
        for j in range(7):
            test_config[j] = lower_limits[j] + random.random() * (upper_limits[j] - lower_limits[j])
        test_configurations.append(test_config)

    num_simple_ik_found, avg_runtime_simple = eval_ik_method(manip, robot, test_configurations,
                                                             simple_ik)
    num_complex_ik_found, avg_runtime_complex = eval_ik_method(manip, robot, test_configurations,
                                                               complex_ik)
    print 'Number of simple IKs found: %i, runtime: %f' % (num_simple_ik_found, avg_runtime_simple)
    print 'Number of complex IKs found: %i, runtime: %f' % (num_complex_ik_found, avg_runtime_complex)
