import openravepy as orpy
import logging
import numpy
from orsampler import RobotCSpaceSampler, GraspApproachConstraintsManager
from sampler import FreeSpaceProximitySampler
from rrt import DynamicPGoalProvider, RRT
from grasp_goal_sampler import GraspGoalSampler
from core import PlanningSceneInterface

class IntegratedHFTSPlanner(object):
    """ Implements a simple to use interface to the integrated HFTS planner. """

    def __init__(self, env_file, hand_file, robot_name, manipulator_name,
                 dof_weights=None, num_hfts_sampling_steps=4,
                 min_iterations=20, max_iterations=70, p_goal_tree=0.8,
                 b_visualize_system=False, b_visualize_grasps=False, b_visualize_hfts=False,
                 free_space_weight=0.1, connected_space_weight=4.0, use_approximates=True,
                 time_limit=60.0):
        """ Creates a new instance of an HFTS planner
            NOTE: It is only possible to display one scene in OpenRAVE at a time. Hence, if the parameters
            b_visualize_system and b_visualize_grasps are both true, only the motion planning scene is shown.
         @param env_file String containing a path to an OpenRAVE environment
         @param hand_file String containing a path to an OpenRAVE hand model (a robot consisting of just the hand)
         @param robot_name String containing the name of the robot to use
         @param manipulator_name String containing the name of the robot's manipulator to use
         @param dof_weights (optional) List of floats, containing a weight for each DOF. These weights are used in
            the distance function on C-Space.
         @param num_hfts_sampling_steps Number of sampling steps for sampling the HFTS.
            (should be greater than HFTS depth)
         @param min_iterations Minimum number of iterations for an optimization step on each HFTS layer
         @param max_iterations Maximum number of iterations for an optimization step on each HFTS layer
         @param p_goal_tree Float denoting the probability of selecting a goal tree rather than an approximate
            goal tree for extensions.
         @param b_visualize_system Boolean, if True, show OpenRAVE viewer displaying the motion planning scene
         @param b_visualize_grasps Boolean, if True, show OpenRAVE viewer displaying the grasp planning scene
         @param b_visualize_hfts Boolean, if True, show a window with a graph visualization of the explored HFTS space
         @param free_space_weight Weight (float) for HFTS rating function t(n) for the distance to
            free space configurations
         @param connected_space_weight Weight (float) for HFTS rating function t(n) for the distance to
            connected configurations
         @param use_approximates Boolean, if True, the grasp sampler returns approximate grasps if it hasn't
            found a valid grasp yet
         @param time_limit Runtime limit for the algorithm in seconds (float)
         """
        self._env = orpy.Environment()
        self._env.Load(env_file)
        if b_visualize_system:
            self._env.SetViewer('qtcoin')
            b_visualize_grasps = False
        if len(self._env.GetRobots()) == 0:
            raise ValueError('The provided environment does not contain a robot!')
        self._robot = self._env.GetRobot(robot_name)
        self._robot.SetActiveManipulator(manipulator_name)
        if dof_weights is None:
            dof_weights = self._robot.GetDOF() * [1.0]
        self._cSampler = RobotCSpaceSampler(self._env, self._robot, scalingFactors=dof_weights)
        # TODO read these robot-specific specs from a file
        planning_scene_interface = PlanningSceneInterface(self._env, self._robot.GetName())
        self._grasp_planner = GraspGoalSampler(hand_path=hand_file,
                                               planning_scene_interface=planning_scene_interface,
                                               visualize=b_visualize_grasps)
        if b_visualize_hfts:
            # TODO: implement this
            raise NotImplementedError('Visualization of HFTS space is not implemented yet.')
        goal_sampler = FreeSpaceProximitySampler(self._grasp_planner, self._cSampler, k=num_hfts_sampling_steps,
                                                 numIterations=max_iterations, minNumIterations=min_iterations,
                                                 returnApproximates=use_approximates,
                                                 connectedWeight=connected_space_weight,
                                                 freeSpaceWeight=free_space_weight)
        self._constraints_manager = GraspApproachConstraintsManager(self._env, self._robot,
                                                                    self._cSampler, numpy.array([0.0, 0.0]))
        p_goal_provider = DynamicPGoalProvider()
        # TODO think about how to make ROS logger run with this
        self._rrt_planner = RRT(p_goal_provider, self._cSampler, goal_sampler, logging.getLogger(),
                                pGoalTree=p_goal_tree, constraintsManager=self._constraints_manager)
        self._time_limit = time_limit

    def load_object(self, obj_file_path, obj_id, obj_id_scene=None):
        if obj_id_scene is None:
            obj_name = obj_id
        else:
            obj_name = obj_id_scene
        self._constraints_manager.set_object_name(obj_name)
        self._grasp_planner.set_object(obj_path=obj_file_path, obj_id=obj_id, obj_id_scene=obj_id_scene)

    def plan(self, start_configuration):
        # TODO remove the following two lines again
        # body = self._env.GetKinBody('test_object')
        # body.Enable(False)
        return self._rrt_planner.proximityBiRRT(start_configuration, timeLimit=self._time_limit)
