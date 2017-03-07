#!/usr/bin/env python

"""This module contains a wrapper class of the HFTS Grasp Sampler."""

from hfts_grasp_planner.core import HFTSSampler, HFTSNode
from sampler import SamplingResult
import logging
import numpy


class HFTSNodeDataExtractor:
    def extractData(self, hierarchyInfo):
        return hierarchyInfo.get_hand_config()

    def getCopyFunction(self):
        return numpy.copy


class GraspGoalSampler:
    """ Wrapper class for the HFTS Grasp Planner/Sampler that allows a full black box usage."""

    # TODO data_path is specific to reading objects from a filesystem. Better probably to pass ObjectIO
    def __init__(self, object_io_interface, hand_path, hand_cache_file, planning_scene_interface,
                 visualize=False, open_hand_offset=0.1):
        """ Creates a new wrapper.
            @param object_io_interface IOObject Object that handles IO requests
            @param hand_path Path to where the hand data is stored.
            @param planning_scene_interface OpenRAVE environment with some additional information
                                            containing the robot and its surroundings.
            @param visualize If true, the internal OpenRAVE environment is set to be visualized
            (only works if there is no other OpenRAVE viewer in this process)
            @param open_hand_offset Value to open the hand by. A grasp is in contact with the target object,
            hence a grasping configuration is always in collision. To enable motion planning to such a
            configuration we open the hand by some constant offset.
            """
        self.grasp_planner = HFTSSampler(object_io_interface=object_io_interface,
                                         vis=visualize, scene_interface=planning_scene_interface)
        self.grasp_planner.set_max_iter(100)
        self.open_hand_offset = open_hand_offset
        self.root_node = self.grasp_planner.get_root_node()
        self.load_hand(hand_path, hand_cache_file)

    def sample(self, depth_limit, post_opt=True):
        """ Samples a grasp from the root level. """
        return self.sample_warm_start(self.root_node, depth_limit, post_opt=post_opt)

    def sample_warm_start(self, hierarchy_node, depth_limit, label_cache=None, post_opt=False):
        """ Samples a grasp from the given node on. """
        logging.debug('[GoalSamplerWrapper] Sampling a grasp from hierarchy depth ' +
                      str(hierarchy_node.get_depth()))
        sampled_node = self.grasp_planner.sample_grasp(node=hierarchy_node, depth_limit=depth_limit,
                                                       post_opt=post_opt,
                                                       label_cache=label_cache,
                                                       open_hand_offset=self.open_hand_offset)
        config = sampled_node.get_arm_configuration()
        if config is not None:
            config = numpy.concatenate((config, sampled_node.get_pre_grasp_config()))
        return SamplingResult(configuration=config, hierarchy_info=sampled_node, data_extractor=HFTSNodeDataExtractor())

    def is_goal(self, sampling_result):
        """ Returns whether the given node is a goal or not. """
        return sampling_result.hierarchyInfo.is_goal()

    def load_hand(self, hand_path, hand_cache_file):
        """ Reset the hand being used. @see __init__ for parameter description. """
        self.grasp_planner.load_hand(hand_file=hand_path, hand_cache_file=hand_cache_file)

    def set_object(self, obj_id, model_id=None):
        """ Set the object.
            @param obj_id String identifying the object.
            @param model_id (optional) Name of the model data. If None, it is assumed to be identical to obj_id
        """
        self.grasp_planner.load_object(obj_id=obj_id, model_id=model_id)
        self.root_node = self.grasp_planner.get_root_node()

    def set_max_iter(self, iterations):
        self.grasp_planner.set_max_iter(iterations)

    def get_max_depth(self):
        return self.grasp_planner.get_maximum_depth()

    def get_root(self):
        return self.root_node

    def set_parameters(self, **kwargs):
        self.grasp_planner.set_parameters(**kwargs)
