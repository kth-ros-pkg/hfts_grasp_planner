import openravepy as orpy
from orsampler import RobotCSpaceSampler

class IntegratedHFTSPlanner(object):
    """ Implements a simple to use interface to the integrated HFTS planner. """
    def __init__(self, env_file, hand_file, robot_file, min_iterations=20, max_iterations=70,
                 b_visualize_system=False, b_visualize_grasps=False, b_visualize_hfts=False,
                 free_space_weight=0.1, connected_space_weight=4.0, use_approximates=True):
        self._env = orpy.Environment()
        self._env.Load(env_file)
        if b_visualize_system:
            self._env.SetViewer('qtcoin')
        self._robot = self._env.GetRobots()[0]
        self._cSampler = RobotCSpaceSampler(self._env, self._robot, scalingFactors=(6 * [1.0]).extend(7 * [0.25]))
        self._grasp_planner = GraspGoalSampler(hand_path=hand_file, or_env=self._env, visualize=b_visualize_grasps,
                                               merged_config=True)
