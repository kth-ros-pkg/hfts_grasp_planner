#!/usr/bin/env python

""" This script contains an OpenRAVE based arm configuration space sampler """

import numpy
import IPython
import random
import openravepy as orpy
import logging
from rrt import SampleData, Constraint, ConstraintsManager
from sampler import CSpaceSampler

NUMERICAL_EPSILON = 0.00001
MINIMAL_STEP_LENGTH = 0.001

class GraspApproachConstraint(Constraint):
    def __init__(self, orEnv, robot, stateSampler, objName, open_hand_config, activationDistance=0.4):
        self.orEnv = orEnv
        self.robot = robot
        self.manip = robot.GetActiveManipulator()
        self.dofIndices = robot.GetActiveDOFIndices()
        self.open_hand_config = open_hand_config
        self.objName = objName
        self.activationDistance = activationDistance
        self.debug = False
        self.stateSampler = stateSampler

    def checkAABBIntersection(self, aabb1, aabb2):
        b_boxes_intersect = True
        mins_1 = aabb1.pos() - aabb1.extents()
        maxs_1 = aabb1.pos() + aabb1.extents()
        mins_2 = aabb2.pos() - aabb2.extents()
        maxs_2 = aabb2.pos() + aabb2.extents()
        for i in range(3):
            b_boxes_intersect = b_boxes_intersect and (mins_1[i] <= maxs_2[i] or mins_2[i] <= maxs_1[i])
        return b_boxes_intersect

    def isActive(self, oldConfig, config):
        with self.orEnv:
            origValues = self.robot.GetDOFValues()
            self.robot.SetDOFValues(oldConfig)
            eefPose = self.manip.GetEndEffectorTransform()
            theObject = self.orEnv.GetKinBody(self.objName)
            objPose = theObject.GetTransform()
            distance = numpy.linalg.norm(eefPose[:3, 3] - objPose[:3, 3])
            self.robot.SetDOFValues(origValues)
            if distance < self.activationDistance:
                return not self.stateSampler.isValid(config)
            return False

    def heuristicGradient(self, config):
        with self.orEnv:
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
            the_object = self.orEnv.GetKinBody(self.objName)
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

    def project(self, oldConfig, config):
        with self.orEnv:
            if self.isActive(oldConfig, config):
                configDir = config - oldConfig
                # logging.debug('[GraspApproachConstraint::project] config: ' + str(config) + ' oldConfig: ' +
                #              str(oldConfig))
                hDir = self.heuristicGradient(oldConfig)
                if hDir is None:
                    logging.warn('[GraspApproachConstraint::project] The heursitic gradient is None')
                    return config
                deltaStep = numpy.dot(configDir, hDir)
                logging.debug('[GraspApproachConstraint::project] Projecting configuration to free' +
                              'space, deltaStep: ' + str(deltaStep))
                if deltaStep <= MINIMAL_STEP_LENGTH: # 0.0:
                    return config
                return oldConfig + deltaStep * hDir
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
        if not tree._bForwardTree:
            # TODO: set activation distance based on object size
            new_constraints.append(GraspApproachConstraint(self.or_env, self.or_robot, self.space_sampler,
                                                           self.object_name, self.open_hand_config))
        self._constraints_storage[tree.getId()] = new_constraints

class RobotCSpaceSampler(CSpaceSampler):
    def __init__(self, orEnv, robot, scalingFactors=None):
        self.orEnv = orEnv
        self.robot = robot
        self.dofIndices = self.robot.GetActiveDOFIndices()
        self.dim = self.robot.GetActiveDOF()
        self.limits = self.robot.GetActiveDOFLimits()
        if scalingFactors is None:
            self._scalingFactors = self.dim * [1]
        else:
            self._scalingFactors = scalingFactors

    def sample(self):
        validSample = False
        randomSample = numpy.zeros(self.dim)
        while not validSample:
            for i in range(self.dim):
                randomSample[i] = random.uniform(self.limits[0][i], self.limits[1][i])
            validSample = self.isValid(randomSample)
        result = SampleData(randomSample)
        return result

    def sampleGaussianNeighborhood(self, config, stdev):
        newConfig = numpy.copy(config)
        for dim in range(len(newConfig)):
            newConfig[dim] = random.gauss(config[dim], stdev)
        return newConfig

    def isValid(self, qcheck):
        inLimits = False
        inCollision = True
        limits = zip(self.limits[0], self.limits[1])
        limitsAndValues = zip(limits, qcheck)
        inLimits = reduce(lambda x, y: x and y,
                          map(lambda tpl: tpl[0][0] - NUMERICAL_EPSILON <= tpl[1] and
                              tpl[1] <= tpl[0][1] + NUMERICAL_EPSILON, limitsAndValues))
        if inLimits:
            with self.orEnv:
                # Save old values
                origValues = self.robot.GetDOFValues()
                activeIndices = self.robot.GetActiveDOFIndices()
                # Set values we wish to test
                self.robot.SetActiveDOFs(self.dofIndices)
                self.robot.SetActiveDOFValues(qcheck)
                # Do collision tests
                inCollision = self.orEnv.CheckCollision(self.robot) or self.robot.CheckSelfCollision()
                # TODO: this function doesn't exist for real openRave robots
                # inCollision = self.robot.CheckCollision()
                # Restore previous state
                self.robot.SetActiveDOFs(activeIndices)
                self.robot.SetJointValues(origValues)
        return inLimits and not inCollision

    def getSamplingStep(self):
        # TODO compute sampling step based on joint value range or sth
        return 0.05

    def getSpaceDimension(self):
        return self.dim

    def getUpperBounds(self):
        return numpy.array(self.robot.GetActiveDOFLimits()[1])

    def getLowerBounds(self):
        return numpy.array(self.robot.GetActiveDOFLimits()[0])

    def getScalingFactors(self):
        return self._scalingFactors

if __name__ == "__main__":
    raise NotImplementedError('Test not implemented')
    # env = orpy.Environment()
    # env.Load('/home/joshua/projects/cvap_kuka_arm/res/kuka-kr5-r850.zae')
    # env.Load('/home/joshua/projects/RRTCostlyGoalRegions/grasping/data/orEnv/cvapRobotLab_objects.xml')
    # sampler = RobotCSpaceSampler(env, KukaSchunkSDH_ORRobot(env.GetRobots()[0]))
    # for i in range(1000):
    #     s = sampler.sample()
    #     print s
