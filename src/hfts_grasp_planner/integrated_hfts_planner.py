import string

import openravepy as orpy
import logging
import numpy
from orsampler import RobotCSpaceSampler, GraspApproachConstraintsManager
from sampler import FreeSpaceProximitySampler
from rrt import DynamicPGoalProvider, RRT
from utils import OpenRAVEDrawer
from grasp_goal_sampler import GraspGoalSampler
from core import PlanningSceneInterface
from hierarchy_visualization import FreeSpaceProximitySamplerVisualizer

class IntegratedHFTSPlanner(object):
    """ Implements a simple to use interface to the integrated HFTS planner. """

    def __init__(self, env_file, hand_file, robot_name, manipulator_name,
                 data_root_path, dof_weights=None, num_hfts_sampling_steps=4,
                 min_iterations=20, max_iterations=70, p_goal_tree=0.8,
                 b_visualize_system=False, b_visualize_grasps=False, b_visualize_hfts=False,
                 b_show_traj=False, b_show_search_tree=False, free_space_weight=0.1, connected_space_weight=4.0,
                 use_approximates=True, compute_velocities=True, time_limit=60.0):
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
         @param b_show_traj Boolean, if True, simulates the trajectory execution in OpenRAVE before returning
         @param b_show_search_tree Boolean, if True, visualizes a projection of the search trees in OpenRAVE
         @param free_space_weight Weight (float) for HFTS rating function t(n) for the distance to
            free space configurations
         @param connected_space_weight Weight (float) for HFTS rating function t(n) for the distance to
            connected configurations
         @param use_approximates Boolean, if True, the grasp sampler returns approximate grasps if it hasn't
            found a valid grasp yet
         @param use_velocities Boolean, if True, compute a whole trajectory (with velocities), else just a path
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
        self._cSampler = RobotCSpaceSampler(self._env, self._robot, scaling_factors=dof_weights)
        # TODO read these robot-specific specs from a file
        planning_scene_interface = PlanningSceneInterface(self._env, self._robot.GetName())
        self._grasp_planner = GraspGoalSampler(data_path=data_root_path,
                                               hand_path=hand_file,
                                               planning_scene_interface=planning_scene_interface,
                                               visualize=b_visualize_grasps)
        hierarchy_visualizer = None
        if b_visualize_hfts:
            hierarchy_visualizer = FreeSpaceProximitySamplerVisualizer(self._robot)
        if num_hfts_sampling_steps <= 0:
            num_hfts_sampling_steps = self._grasp_planner.get_max_depth() + 1
        self._hierarchy_sampler = FreeSpaceProximitySampler(self._grasp_planner, self._cSampler, k=num_hfts_sampling_steps,
                                                            num_iterations=max_iterations, min_num_iterations=min_iterations,
                                                            b_return_approximates=use_approximates,
                                                            connected_weight=connected_space_weight,
                                                            free_space_weight=free_space_weight,
                                                            debug_drawer=hierarchy_visualizer)
        # TODO the open hand configuration should be given from a configuration file
        self._constraints_manager = GraspApproachConstraintsManager(self._env, self._robot,
                                                                    self._cSampler, numpy.array([0.0, 0.0495]))
        p_goal_provider = DynamicPGoalProvider()
        self._debug_tree_drawer = None
        if b_show_search_tree:
            self._debug_tree_drawer = OpenRAVEDrawer(self._env, self._robot, True)
        self._rrt_planner = RRT(p_goal_provider, self._cSampler, self._hierarchy_sampler, logging.getLogger(),
                                pgoal_tree=p_goal_tree, constraints_manager=self._constraints_manager)
        self._time_limit = time_limit
        self._last_path = None
        self._compute_velocities = compute_velocities
        self._b_show_trajectory = b_show_traj

    def load_object(self, obj_id, model_id=None):
        self._constraints_manager.set_object_name(obj_id)
        self._grasp_planner.set_object(obj_id=obj_id, model_id=model_id)

    def get_robot(self):
        return self._robot

    def create_or_trajectory(self, path, vel_factor=0.2):
        if path is None:
            return None
        configurations_path = map(lambda x: x.get_configuration(), path)
        # The path ends in a pre-grasp configuration.
        # The final grasp configuration is stored as additional data in the last waypoint,
        # so we need to construct the final configuration here.
        grasp_hand_config = path[-1].get_data()
        last_config = numpy.array(configurations_path[-1])
        hand_idxs = self._robot.GetActiveManipulator().GetGripperIndices()
        assert len(hand_idxs) == len(grasp_hand_config)
        j = 0
        for i in hand_idxs:
            last_config[i] = grasp_hand_config[j]
            j += 1
        configurations_path.append(last_config)

        vel_limits = self._robot.GetDOFVelocityLimits()
        self._robot.SetDOFVelocityLimits(vel_factor * vel_limits)
        traj = orpy.RaveCreateTrajectory(self._env, '')
        cs = traj.GetConfigurationSpecification()
        dof_string = string.join([' ' + str(x) for x in range(self._robot.GetDOF())])
        cs.AddGroup('joint_values ' + self._robot.GetName() + dof_string, self._robot.GetDOF(), 'linear')
        # cs.AddDerivativeGroups(1, True)
        traj.Init(cs)
        for idx in range(len(configurations_path)):
            traj.Insert(idx, configurations_path[idx])
        orpy.planningutils.RetimeTrajectory(traj, hastimestamps=False)
        self._robot.SetDOFVelocityLimits(vel_limits)
        return traj

    def plan(self, start_configuration):
        if self._debug_tree_drawer is not None:
            self._debug_tree_drawer.clear()
            debug_function = self._debug_tree_drawer.draw_trees
        else:
            def debug_function(forward_tree, backward_trees):
                pass
        self._last_path = self._rrt_planner.proximity_birrt(start_configuration, time_limit=self._time_limit,
                                                            debug_function=debug_function)
        if self._compute_velocities:
            self._last_traj = self.create_or_trajectory(self._last_path)
            if self._b_show_trajectory and self._last_traj is not None:
                controller = self._robot.GetController()
                controller.SetPath(self._last_traj)
                self._robot.WaitForController(self._last_traj.GetDuration())
                controller.Reset()
            return self._last_traj
        return self._last_path

    def set_parameters(self, min_iterations=None, max_iterations=None,
                       free_space_weight=None, connected_space_weight=None,
                       use_approximates=None, compute_velocities=None,
                       time_limit=None, com_center_weight=None,
                       pos_reach_weight=None, f01_parallelism_weight=None,
                       grasp_symmetry_weight=None, grasp_flatness_weight=None,
                       reachability_weight=None, hfts_generation_params=None,
                       b_force_new_hfts=None):
        # TODO some of these parameters are robot hand specific
        if time_limit is not None:
            self._time_limit = time_limit
        if compute_velocities is not None:
            self._compute_velocities = compute_velocities
        self._grasp_planner.set_parameters(com_center_weight=com_center_weight,
                                           pos_reach_weight=pos_reach_weight,
                                           f01_parallelism_weight=f01_parallelism_weight,
                                           grasp_symmetry_weight=grasp_symmetry_weight,
                                           grasp_flatness_weight=grasp_flatness_weight,
                                           reachability_weight=reachability_weight,
                                           b_force_new_hfts=b_force_new_hfts,
                                           hfts_generation_params=hfts_generation_params)
        self._hierarchy_sampler.set_parameters(min_iterations=min_iterations,
                                               max_iterations=max_iterations,
                                               free_space_weight=free_space_weight,
                                               connected_space_weight=connected_space_weight,
                                               use_approximates=use_approximates)
        # TODO implement the rest

