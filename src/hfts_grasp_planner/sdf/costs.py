"""
    This module provides a selection of SDF-based cost functions for robot configurations.
"""
import numpy as np

class DistanceToFreeSpace(object):
    """
        Not really a cost, but this class provides a function that maps a robot configuration
        to a distance value > 0, if the robot configuration collides with obstacles and to 0, if
        it is collision free. While it behaves similar to a distance to free space,
        the unit of the distance is in meters and not in radians.
    """
    def __init__(self, robot, robot_sdf, safety_margin=0.01):
        """
            Creates a new DistanceToFreeSpace function for the specified robot.
            @param robot - OpenRAVE robot to use
            @param robot_sdf - RobotSDF for this robot
        """
        self._robot = robot
        self._sdf = robot_sdf
        self._safety_margin = safety_margin

    def get_distance_to_free_space(self, config=None):
        """
            Returns a distance of the current / or the provided configuration
            to freespace.
            @param config - if provided, the distance for this configuration is returned,
                            else the distance for the current configuration of the robot
        """
        original_config = self._robot.GetDOFValues()
        if config is not None:
            self._robot.SetDOFValues(config)
        distances = self._sdf.get_distances()
        distances -= self._safety_margin
        return -1.0 * sum(np.clip(distances, a_min=None, a_max=0.0))
