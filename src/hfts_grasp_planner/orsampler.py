#!/usr/bin/env python

""" This script contains an OpenRAVE based arm configuration space sampler """

import numpy
import random
import openravepy as orpy
import logging
from rrt import SampleData, Constraint, ConstraintsManager
from sampler import CSpaceSampler

NUMERICAL_EPSILON = 0.00001
MINIMAL_STEP_LENGTH = 0.001


class GraspApproachConstraint(Constraint):
    def __init__(self, or_env, robot, state_sampler, obj_name, open_hand_config, activation_distance=0.4):
        self.or_env = or_env
        self.robot = robot
        self.manip = robot.GetActiveManipulator()
        self.dof_indices = robot.GetActiveDOFIndices()
        self.open_hand_config = open_hand_config
        self.obj_name = obj_name
        self.activation_distance = activation_distance
        self.debug = False
        self.state_sampler = state_sampler

    def check_aabb_intersection(self, aabb1, aabb2):
        b_boxes_intersect = True
        mins_1 = aabb1.pos() - aabb1.extents()
        maxs_1 = aabb1.pos() + aabb1.extents()
        mins_2 = aabb2.pos() - aabb2.extents()
        maxs_2 = aabb2.pos() + aabb2.extents()
        for i in range(3):
            b_boxes_intersect = b_boxes_intersect and (mins_1[i] <= maxs_2[i] or mins_2[i] <= maxs_1[i])
        return b_boxes_intersect

    def is_active(self, old_config, config):
        with self.or_env:
            orig_values = self.robot.GetDOFValues()
            self.robot.SetDOFValues(old_config)
            eef_pose = self.manip.GetEndEffectorTransform()
            the_object = self.or_env.GetKinBody(self.obj_name)
            obj_pose = the_object.GetTransform()
            distance = numpy.linalg.norm(eef_pose[:3, 3] - obj_pose[:3, 3])
            self.robot.SetDOFValues(orig_values)
            if distance < self.activation_distance:
                return not self.state_sampler.is_valid(config)
            return False

    def heuristic_gradient(self, config):
        with self.or_env:
            old_values = self.robot.GetDOFValues()
            # Set the old configuration
            self.robot.SetDOFValues(config)
            # First compute hand configuration gradient
            hand_dofs = self.manip.GetGripperIndices()
            old_hand_config = self.robot.GetDOFValues(hand_dofs)
            hand_open_dir = self.open_hand_config - old_hand_config
            hand_open_dir_magn = numpy.linalg.norm(hand_open_dir)
            # TODO we might wanna replace this with manip.GetClosingDirection
            if hand_open_dir_magn > 0.0:
                hand_open_dir = 1.0 / hand_open_dir_magn * hand_open_dir
                hand_gradient = 0.1 * hand_open_dir
            else:
                hand_gradient = len(hand_dofs) * [0.0]
            # Now compute the arm configuration gradient
            the_object = self.or_env.GetKinBody(self.obj_name)
            obj_pose = the_object.GetTransform()
            eef_pose = self.manip.GetEndEffectorTransform()
            inv_approach_dir = eef_pose[:3, 3] - obj_pose[:3, 3]
            distance = numpy.linalg.norm(inv_approach_dir)
            if distance == 0.0:
                self.robot.SetDOFValues(old_values)
                return None
            inv_approach_dir = 1.0 / distance * inv_approach_dir
            jacobian = self.manip.CalculateJacobian()
            pseudo_inverse = numpy.linalg.pinv(jacobian)
            arm_gradient = numpy.dot(pseudo_inverse, inv_approach_dir)
            gradient = numpy.concatenate((arm_gradient, hand_gradient))
            self.robot.SetDOFValues(old_values)
            return 1.0 / numpy.linalg.norm(gradient) * gradient

    def project(self, old_config, config):
        with self.or_env:
            if self.is_active(old_config, config):
                config_dir = config - old_config
                # logging.debug('[GraspApproachConstraint::project] config: ' + str(config) + ' oldConfig: ' +
                #              str(oldConfig))
                h_dir = self.heuristic_gradient(old_config)
                if h_dir is None:
                    logging.warn('[GraspApproachConstraint::project] The heuristic gradient is None')
                    return config
                delta_step = numpy.dot(config_dir, h_dir)
                logging.debug('[GraspApproachConstraint::project] Projecting configuration to free' +
                              'space, delta_step: ' + str(delta_step))
                if delta_step <= MINIMAL_STEP_LENGTH: # 0.0:
                    return config
                return old_config + delta_step * h_dir
            # logging.debug('[GraspApproachConstraint::project] Not active')
            return config


class GraspApproachConstraintsManager(ConstraintsManager):
    def __init__(self, or_env, robot, space_sampler, open_hand_config):
        super(GraspApproachConstraintsManager, self).__init__()
        self.or_env = or_env
        self.or_robot = robot
        self.object_name = None
        self.space_sampler = space_sampler
        self.open_hand_config = open_hand_config

    def set_object_name(self, object_name):
        self.object_name = object_name

    def register_new_tree(self, tree):
        new_constraints = []
        # Except for the forward tree, create GraspApproachConstraints
        if not tree._b_forward_tree:
            # TODO: set activation distance based on object size
            new_constraints.append(GraspApproachConstraint(self.or_env, self.or_robot, self.space_sampler,
                                                           self.object_name, self.open_hand_config))
        self._constraints_storage[tree.get_id()] = new_constraints


class RobotCSpaceSampler(CSpaceSampler):
    def __init__(self, or_env, robot, scaling_factors=None):
        self.or_env = or_env
        self.robot = robot
        self.dof_indices = self.robot.GetActiveDOFIndices()
        self.dim = self.robot.GetActiveDOF()
        self.limits = self.robot.GetActiveDOFLimits()
        if scaling_factors is None:
            self._scaling_factors = self.dim * [1]
        else:
            self._scaling_factors = scaling_factors

    def sample(self):
        valid_sample = False
        random_sample = numpy.zeros(self.dim)
        while not valid_sample:
            for i in range(self.dim):
                random_sample[i] = random.uniform(self.limits[0][i], self.limits[1][i])
            valid_sample = self.is_valid(random_sample)
        result = SampleData(random_sample)
        return result

    def sample_gaussian_neighborhood(self, config, stdev):
        new_config = numpy.copy(config)
        for dim in range(len(new_config)):
            new_config[dim] = random.gauss(config[dim], stdev)
        return new_config

    def is_valid(self, qcheck):
        in_limits = False
        in_collision = True
        limits = zip(self.limits[0], self.limits[1])
        limits_and_values = zip(limits, qcheck)
        in_limits = reduce(lambda x, y: x and y,
                           map(lambda tpl: tpl[0][0] - NUMERICAL_EPSILON <= tpl[1] <= tpl[0][1] + NUMERICAL_EPSILON,
                               limits_and_values))
        if in_limits:
            with self.or_env:
                # Save old values
                orig_values = self.robot.GetDOFValues()
                active_indices = self.robot.GetActiveDOFIndices()
                # Set values we wish to test
                self.robot.SetActiveDOFs(self.dof_indices)
                self.robot.SetActiveDOFValues(qcheck)
                # Do collision tests
                in_collision = self.or_env.CheckCollision(self.robot) or self.robot.CheckSelfCollision()
                # Restore previous state
                self.robot.SetActiveDOFs(active_indices)
                self.robot.SetJointValues(orig_values)
        return in_limits and not in_collision

    def get_sampling_step(self):
        # TODO compute sampling step based on joint value range or sth
        return 0.05

    def get_space_dimension(self):
        return self.dim

    def get_upper_bounds(self):
        return numpy.array(self.robot.GetActiveDOFLimits()[1])

    def get_lower_bounds(self):
        return numpy.array(self.robot.GetActiveDOFLimits()[0])

    def get_scaling_factors(self):
        return self._scaling_factors

if __name__ == "__main__":
    raise NotImplementedError('Test not implemented')
    # env = orpy.Environment()
    # env.Load('/home/joshua/projects/cvap_kuka_arm/res/kuka-kr5-r850.zae')
    # env.Load('/home/joshua/projects/RRTCostlyGoalRegions/grasping/data/orEnv/cvapRobotLab_objects.xml')
    # sampler = RobotCSpaceSampler(env, KukaSchunkSDH_ORRobot(env.GetRobots()[0]))
    # for i in range(1000):
    #     s = sampler.sample()
    #     print s
