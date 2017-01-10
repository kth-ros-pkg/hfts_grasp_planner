#!/usr/bin/env python

"""This module contains a wrapper class of the HFTS Grasp Sampler."""

from hfts_grasp_planner.core import HFTSSampler, HFTSNode
from sampler import SamplingResult, HierarchyNode
import logging
import numpy


class HFTSNodeDataExtractor:
    def extractData(self, hierarchyInfo):
        return hierarchyInfo.getHandConfig()

    def getCopyFunction(self):
        return numpy.copy


class GraspGoalSampler:
    """ Wrapper class for the HFTS Grasp Planner/Sampler that allows a full black box usage."""

    def __init__(self, hand_path, or_env,
                 visualize=False, open_hand_offset=0.1):
        """ Creates a new wrapper.
            @param hand_path Path to where the hand data is stored.
            @param or_env OpenRAVE environment containing the robot and its surroundings.
            @param visualize If true, the internal OpenRAVE environment is set to be visualized
            (only works if there is no other OpenRAVE viewer in this process)
            @param open_hand_offset Value to open the hand by. A grasp is in contact with the target object,
            hence a grasping configuration is always in collision. To enable motion planning to such a
            configuration we open the hand by some constant offset.
            """
        self.grasp_planner = HFTSSampler(vis=visualize)
        self.grasp_planner.setMaxIter(100)
        self.open_hand_offset = open_hand_offset
        self.root_node = self.grasp_planner.getRootNode()

    def sample(self, depth_limit, post_opt=True):
        """ Samples a grasp from the root level. """
        return self.sample_warm_start(self.root_node, depth_limit, post_opt=post_opt)

    def sample_warm_start(self, hierarchy_node, depth_limit, label_cache=None, post_opt=False):
        """ Samples a grasp from the given node on. """
        logging.debug('[GoalSamplerWrapper] Sampling a grasp from hierarchy depth ' +
                      str(hierarchy_node.getDepth()))
        sampled_node = self.grasp_planner.sampleGrasp(node=hierarchy_node, depthLimit=depth_limit,
                                                      postOpt=post_opt,
                                                      labelCache=label_cache,
                                                      openHandOffset=self.open_hand_offset)
        config = sampled_node.getArmConfig()
        if self.b_merged_config and config is not None:
            config = numpy.concatenate((config, sampled_node.getPreGraspHandConfig()))
        return SamplingResult(configuration=config, hierarchyInfo=sampled_node, dataExtractor=HFTSNodeDataExtractor())

    def isGoal(self, sampling_result):
        """ Returns whether the given node is a goal or not. """
        return sampling_result.hierarchyInfo.isGoal()

    def resetHand(self, hand_path):
        """ Reset the hand being used. @see __init__ for parameter description. """
        self.grasp_planner.loadHand(dataPath=hand_path)

    def resetObject(self, obj_path, obj_id):
        """ Reset the object and the environment.
            @param obj_path String containing the path to  """
        self.grasp_planner.loadObj(dataPath=obj_path, objName=obj_id)
        # self.grasp_planner.loadLabEnv(env=or_env, vis=False, loadLS=False)
        self.grasp_planner.computeContactCombinations()
        self.root_node = self.grasp_planner.getRootNode()

    def setMaxIter(self, iterations):
        self.grasp_planner.setMaxIter(iterations)

    def getMaxDepth(self):
        return self.grasp_planner.getMaximumDepth()

    def getRoot(self):
        return self.root_node
