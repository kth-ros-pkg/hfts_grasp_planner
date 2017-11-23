"""
    This module provides SDF-based information for an OpenRAVE robot. In particular,
    this module provides an SDF-based obstacle cost function.
"""
import yaml
import numpy as np
# from . import core

class RobotSDF(object):
    """
        This class provides signed distance information for a robot.
        It utilizes a set of balls to approximate a robot and provides access functions
        to acquire the shortest distance of each ball to the closest obstacle.
        In order to instantiate an object of this class, you need a signed distance field
        for the OpenRAVE scene the robot is embedded in as well as description file
        containing the definition of the approximating balls.
    """
    def __init__(self, robot, scene_sdf):
        """
            Creates a new RobotSDF.
            NOTE: Before you can use this object, you need to provide it with a ball approximation.
            @param robot - OpenRAVE robot this sdf should operate on
            @param scene_sdf - A SceneSDF for the given environment.
        """
        self._robot = robot
        self._sdf = scene_sdf
        self._handles = []  # list of openrave handles for visualization
        # ball_positoins stores one large matrix of shape (n, 4), where n is the number of balls we have
        # in each row we have the homogeneous coordinates of on ball center w.r.t its link's frame
        self._ball_positions = None
        self._query_positions = None
        # ball_radii stores the radii for all balls
        self._ball_radii = None
        # saves the indices of links for which we have balls
        self._link_indices = None
        # saves tuples (num_balls, index_offset) for each link_idx in self._link_indices
        # index_offset refers to the row in self._ball_positions and ball_radii in which the first ball
        # for link i is stored. num_balls is the number of balls for this link
        self._ball_indices = None

    def load_approximation(self, filename):
        """
            Loads a ball approximation for the robot from the given file.
        """
        self._ball_positions = []
        self._ball_radii = []
        self._link_indices = []
        link_descs = None
        with open(filename, 'r') as in_file:
            link_descs = yaml.load(in_file)
        # first we need to know how many balls we have
        num_balls = 0
        for ball_descs in link_descs.itervalues():
            num_balls += len(ball_descs)
        # now we can create our data structures
        self._ball_positions = np.ones((num_balls, 4))
        self._query_positions = np.ones((num_balls, 4))
        self._ball_radii = np.zeros(num_balls)
        self._ball_indices = []
        index_offset = 0
        # run over all links
        links = self._robot.GetLinks()
        for link_idx in range(len(links)):  # we need the index
            link_name = links[link_idx].GetName()
            if link_name in link_descs:  # if we have some balls for this link
                ball_descs = link_descs[link_name]
                num_balls_link = len(ball_descs)
                index_offset_link = index_offset
                # save this link
                self._link_indices.append(link_idx)
                # save the offset and number of balls
                self._ball_indices.append((num_balls_link, index_offset_link))
                # save all balls
                for ball_idx in range(num_balls_link):
                    self._ball_positions[index_offset_link + ball_idx, :3] = np.array(ball_descs[ball_idx][:3])
                    self._ball_radii[index_offset_link + ball_idx] = ball_descs[ball_idx][3]
                index_offset += num_balls_link
            else:
                # for links that don't have any balls we need to store None so we can index ball_indices easily
                self._ball_indices.append(None)

    def get_distances(self):
        """
            Returns the distances of all balls to the respective closest obstacle.
        """
        if self._ball_positions is None:
            return None
        link_tfs = self._robot.GetLinkTransformations()
        for link_idx in self._link_indices:
            nb, off = self._ball_indices[link_idx]  # number of balls, offset
            self._query_positions[off:off + nb] = np.dot(self._ball_positions[off:off + nb], link_tfs[link_idx].transpose())
        return self._sdf.get_distances(self._query_positions) - self._ball_radii

    def visualize_balls(self):
        """
            If approximating balls are available, this function issues rendering these
            in OpenRAVE.
        """
        self.hide_balls()
        if self._ball_positions is None:
            return
        env = self._robot.GetEnv()
        link_tfs = self._robot.GetLinkTransformations()
        color = np.array((1, 0, 0, 0.6))
        for link_idx in self._link_indices:
            (nb, off) = self._ball_indices[link_idx]  # number of balls, offest
            positions = np.dot(self._ball_positions[off:off + nb], link_tfs[link_idx].transpose())
            for ball_idx in range(nb):
                handle = env.plot3(positions[ball_idx, :3], self._ball_radii[off + ball_idx],
                                   color, 1)
                self._handles.append(handle)

    def hide_balls(self):
        """
            If the approximating balls are currently rendered, this function
            removes those renderings, else it does nothing.
        """
        self._handles = []
