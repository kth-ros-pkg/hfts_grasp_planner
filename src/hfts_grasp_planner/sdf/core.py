"""
    This module contains the core components for signed distance fields.
"""
from __future__ import print_function
import math
import time
import operator
import itertools
import skfmm
import numpy as np
import openravepy as orpy


class VoxelGrid(object):
    """"
        A voxel grid is a 3D discretization of a robot's workspace.
        For each voxel in this grid, this voxel grid saves a single floating point number.
    """
    class VoxelCell(object):
        """
            A voxel cell is a cell in a voxel grid and represents a voxel.
        """
        def __init__(self, grid, idx):
            self._grid = grid
            self._idx = idx

        def get_idx(self):
            """
                Return the index of this cell.
            """
            return self._idx

        def get_position(self):
            """
                Return the position in R^3 of the center of this cell.
            """
            return self._grid.get_cell_position(self._idx)

        def get_size(self):
            """
                Returns the edge length of this cell.
            """
            return self._grid.get_cell_size()

        def get_value(self):
            """
                Return the value stored in this cell
            """
            return self._grid.get_cell_value(self._idx)

        def set_value(self, value):
            """
                Set the value of this cell
            """
            self._grid.set_cell_value(self._idx, value)


    def __init__(self, workspace_aabb, cell_size=0.02, base_pos=None):
        """
            Creates a new voxel grid covering the specified workspace volume.
            @param workspace_aabb - bounding box of the workspace as numpy array of form
                                    [min_x, min_y, min_z, max_x, max_y, max_z]
            @param cell_size - cell size of the voxel grid (in meters)
            @param base_pos - if not None, any query point is shifted by base_pos
        """
        # TODO there is still a bug here regarding the bse position
        self._cell_size = cell_size
        self._dimensions = workspace_aabb[3:] - workspace_aabb[:3]
        self._num_cells = [int(math.ceil(x)) for x in self._dimensions / cell_size]
        self._base_pos = workspace_aabb[:3]
        if base_pos is not None:
            self._base_pos += base_pos
        self._cells = np.zeros(self._num_cells)

    def __iter__(self):
        return self.get_cell_generator()

    def get_index_generator(self):
        """
            Returns a generator that generates all indices of this grid.
        """
        return ((ix, iy, iz) for ix in xrange(self._num_cells[0])
                for iy in xrange(self._num_cells[1])
                for iz in xrange(self._num_cells[2]))

    def get_cell_generator(self):
        """
            Returns a generator that generates all cells in this grid
        """
        index_generator = self.get_index_generator()
        return (VoxelGrid.VoxelCell(self, idx) for idx in index_generator)

    def get_cell_idx(self, pos):
        """
            Returns the index triple of the voxel in which the specified position lies
        """
        rel_pos = pos - self._base_pos
        return [int(math.floor(x / self._cell_size)) for x in rel_pos]

    def get_cell_position(self, idx, b_center=True):
        """
            Returns the position in R^3 of the center or min corner of the voxel with index idx
            @param idx - a tuple/list of length 3 (ix, iy, iz) specifying the voxel
            @param b_center - if true, it return the position of the center, else of min corner
            @return numpy.array representing the center or min corner position of the voxel
        """
        rel_pos = np.array(idx) * self._cell_size
        if b_center:
            rel_pos += np.array([self._cell_size / 2.0, self._cell_size / 2.0, self._cell_size / 2.0])
        return self._base_pos + rel_pos

    def get_cell_value(self, idx):
        """
            Returns the value of the specified cell
        """
        idx = self.sanitize_idx(idx)
        return self._cells[idx[0], idx[1], idx[2]]

    def get_num_cells(self):
        """
            Returns the number of cells this grid has in each dimension.
            @return (nx, ny, nz)
        """
        return tuple(self._num_cells)

    def get_raw_data(self):
        """
            Returns a reference to the underlying cell data structure.
            Use with caution!
        """
        return self._cells

    def set_raw_data(self, data):
        """
            Overwrites the underlying cell data structure with the provided one.
            Use with caution!
            @param data - a numpy array with the same shape as returned by get_raw_data()
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('The type of the provided data is invalid. Must be numpy.ndarray, but it is %s' % str(type(data)))
        if data.shape != self._cells.shape:
            raise ValueError("The shape of the provided data differs from this grid's shape." +
                             " Input shape is %s, required shape %s" % (str(data.shape), str(self._cells.shape)))
        self._cells = data

    def get_cell_size(self):
        """
            Returns the cell size
        """
        return self._cell_size

    def sanitize_idx(self, idx):
        """
            Ensures that the provided index is a valid index type.
        """
        if len(idx) != 3:
            raise ValueError("Provided index has invalid length (%i)" % len(idx))
        return map(int, idx)

    def set_cell_value(self, idx, value):
        """
            Sets the value of the cell with given index.
            @param idx - tuple (ix, iy, iz)
            @param value - value to set the cell to
        """
        idx = self.sanitize_idx(idx)
        self._cells[idx[0], idx[1], idx[2]] = value

    def fill(self, min_idx, max_idx, value):
        """
            Fills all cells in the block min_idx, max_idx with value
        """
        min_idx = self.sanitize_idx(min_idx)
        max_idx = self.sanitize_idx(max_idx)
        self._cells[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] = value

    def get_max_value(self):
        """
            Returns the maximal value in this grid.
        """
        values = [x.get_value() for x in self]
        return max(values)

    def get_min_value(self):
        """
            Returns the minimal value in this grid.
        """
        values = [x.get_value() for x in self]
        return min(values)


class ORVoxelGridVisualization(object):
    """
        This class allows to visualize a voxel grid using an OpenRAVE environment.
    """
    def __init__(self, or_env, voxel_grid):
        """
            Creates a new visualization of a voxel grid using openrave.
        """
        self._env = or_env
        self._voxel_grid = voxel_grid
        self._handles = []

    def update(self, min_sat_value=None, max_sat_value=None):
        """
            Updates this visualization to reflect the latest state of the underlying voxel grid.
            The voxels are colored according to their values. By default the color of a voxel
            is computed using linear interpolation between two colors min_color and max_color.
            The voxel with maximum value is colored with max_color and the voxel with minimum value
            is colored with min_color. This behaviour can be changed by providing min_sat_value
            and max_sat_value. If these values are provided, any cell with value <= min_sat_value is
            colored min_color and any cell with value >= max_sat_value is colored with max_color.
            Cells with values in range (min_sat_value, max_sat_value) are colored using linear interpolation.

            @param min_sat_value (optional) - minimal saturation value
            @param max_sat_value (optional) - maximal saturation value
        """
        self._handles = []
        values = [x.get_value() for x in self._voxel_grid]
        if min_sat_value is None:
            min_sat_value = min(values)
        if max_sat_value is None:
            max_sat_value = max(values)

        blue_color = np.array([0.0, 0.0, 1.0, 0.1])
        red_color = np.array([1.0, 0.0, 0.0, 0.1])
        positions = np.array([cell.get_position() for cell in self._voxel_grid])
        def compute_color(value):
            """
                Computes the color for the given value
            """
            rel_value = np.clip((value - min_sat_value) / (max_sat_value - min_sat_value), 0.0, 1.0)
            return (1.0 - rel_value) * red_color + rel_value * blue_color
        colors = np.array([compute_color(v) for v in values])
        # TODO we should read the conversion from pixels to workspace size from somwhere
        # and convert true cell size to it
        handle = self._env.plot3(positions, 20, colors)  # size is in pixel
        self._handles.append(handle)


class SDF(object):
    """
        This class provides a workspace signed distance field for a robot in some environment.
    """
    class BodyManager(object):
        """
            Internal helper class for creating binary collision maps
        """
        def __init__(self, env, cell_size):
            """
                Create a new BodyManager
                @param env - OpenRAVE environment
                @param cell_size - size of a cell
            """
            self._env = env
            self._cell_size = cell_size
            self._bodies = {}
            self._active_body = None

        def get_body(self, dimensions):
            """
                Get a kinbody that covers the given number of cells.
                @param dimensions - numpy array (wx, wy, wz)
            """
            new_active_body = None
            if tuple(dimensions) in self._bodies:
                new_active_body = self._bodies[tuple(dimensions)]
            else:
                new_active_body = orpy.RaveCreateKinBody(self._env, '')
                new_active_body.SetName("CollisionCheckBody" + str(dimensions[0]) + str(dimensions[1]) + str(dimensions[2]))
                physical_dimensions = self._cell_size * dimensions
                new_active_body.InitFromBoxes(np.array([[0, 0, 0,
                                                 physical_dimensions[0] / 2.0,
                                                 physical_dimensions[1] / 2.0,
                                                 physical_dimensions[2] / 2.0]]),
                                        True)
                self._env.AddKinBody(new_active_body)
                self._bodies[tuple(dimensions)] = new_active_body
            if new_active_body is not self._active_body and self._active_body is not None:
                self._active_body.Enable(False)
                self._active_body.SetVisible(False)
            self._active_body = new_active_body
            self._active_body.Enable(True)
            self._active_body.SetVisible(True)
            return self._active_body

        def clear(self):
            for body in self._bodies.itervalues():
                self._env.Remove(body)
                body.Destroy()
            self._bodies = {}


    def __init__(self, env_path, robot_name=None, manip_name=None):
        """
            Creates a new signed distance field for the specified environment.
            @param env_path - path to the environment
            @param robot_name(optional) - name of the robot
            @param manip_name(optional) - name of the manipulator for which the sdf is created TODO CURRENTLY NOT SUPPORTED
        """
        self._env = orpy.Environment()
        b_env_loaded = self._env.Load(env_path)
        if not b_env_loaded:
            raise IOError('Could not create signed distance field, because the'
                          ' environment %s can not be loaded' % env_path)
        if robot_name:
            self._robot = self._env.GetRobot(robot_name)
        else:
            robots = self._env.GetRobots()
            if robots:
                self._robot = robots[0]
            else:
                self._robot = None
        if self._robot is None:
            raise ValueError('Could not create signed distance field. The environment does not contain any robot or the requested robot')
        # if manip_name:
        #     self._manipulator = self._robot.GetManipulator(manip_name)
        # else:
        #     self._manipulator = self._robot.GetActiveManipulator()
        # if not self._manipulator:
        #     raise ValueError('Could not create signed distance field. The robot has no manipulator or the requested manipulator does not exist')
        self._grid = None
        self._or_visualization = None

    def __del__(self):
        if self._env:
            self._env.Destroy()
        self._env = None

    def _check_cell_collision(self, cell):
        tranform = self._cell_body.GetTransform()
        tranform[0:3, 3] = cell.get_position()
        self._cell_body.SetTransform(tranform)
        b_collision = self._env.CheckCollision(self._cell_body)
        if b_collision:
            cell.set_value(-1.0)
        else:
            cell.set_value(1.0)
        self._counter += 1
        print("Covered %i / %i cells" % (self._counter, reduce(operator.mul, self._grid.get_num_cells())))

    def _compute_bcm_rec(self, min_idx, max_idx, body_manager, covered_volume):
        """
            Computes a binary collision map recursively.
            INVARIANT: This function is only called if there is a collision for a box ranging from min_idx to max_idx
            @param min_idx - numpy array [min_x, min_y, min_z] cell indices
            @param max_idx - numpy array [max_x, max_y, max_z] cell indices (the box excludes these)
            @param body_manager - a body manager of type BodyManager
        """
        # Base case, we are looking at only one cell
        if (min_idx + 1 == max_idx).all():
            self._grid.set_cell_value(min_idx, -1.0)
            return covered_volume + 1
        # else we need to split this cell up and see which child ranges are in collision
        box_size = max_idx - min_idx  # the number of cells along each axis in this box
        half_sizes = np.zeros((2, 3))
        half_sizes[0] = map(math.floor, box_size / 2)  # we split this box into 8 children by dividing along each axis
        half_sizes[1] = box_size - half_sizes[0]  # half_sizes stores the divisions for each axis
        # now we create the actual ranges for each of the 8 children
        children_dimensions = itertools.product(half_sizes[:, 0], half_sizes[:, 1], half_sizes[:, 2])
        # and the position offsets
        offset_matrix = np.zeros((2, 3))
        offset_matrix[1] = half_sizes[0]
        rel_min_indices = itertools.product(offset_matrix[:, 0], offset_matrix[:, 1], offset_matrix[:, 2])
        for (rel_min_idx, child_dim) in itertools.izip(rel_min_indices, children_dimensions):
            volume = reduce(operator.mul, child_dim)
            if volume != 0:
                child_min_idx = min_idx + np.array(rel_min_idx)
                child_max_idx = child_min_idx + np.array(child_dim)
                child_physical_dimensions = self._grid.get_cell_size() * np.array(child_dim)
                cell_body = body_manager.get_body(np.array(child_dim))
                transform = cell_body.GetTransform()
                transform[0:3, 3] = self._grid.get_cell_position(child_min_idx, b_center=False)
                transform[0:3, 3] += child_physical_dimensions / 2.0  # the center of our big box
                cell_body.SetTransform(transform)
                if self._env.CheckCollision(cell_body):
                    covered_volume = self._compute_bcm_rec(child_min_idx, child_max_idx, body_manager, covered_volume)
                else:
                    self._grid.fill(child_min_idx, child_max_idx, 1.0)
                    covered_volume += volume
        # total_volme = reduce(operator.mul, self._grid.get_num_cells())
        # print("Covered %i / %i cells" % (covered_volume, total_volme))
        return covered_volume

    def _compute_bcm(self):
        # compute for each cell whether it collides with anything
        self._robot.Enable(False)
        # self._cell_body = orpy.RaveCreateKinBody(self._env, '')
        # self._cell_body.SetName("CollisionCheckBody")
        # self._env.AddKinBody(self._cell_body)
        # self._counter = 0
        # map(self._check_cell_collision, self._grid)
        body_manager = SDF.BodyManager(self._env, self._grid.get_cell_size())
        self._compute_bcm_rec(np.array([0, 0, 0]), self._grid.get_num_cells(), body_manager, 0)
        self._robot.Enable(True)
        body_manager.clear()
        # Finally remove the collision body we used again
        # self._env.Remove(self._cell_body)
        # self._cell_body.Destroy()
        # self._cell_body = None

    def _compute_sdf(self):
        # TODO find a good solution for the problem that we get an exception if there are no collisions at all
        min_value = self._grid.get_min_value()
        if min_value > 0:
            self._grid.set_cell_value((0, 0, 0), -1.0)
        self._grid.set_raw_data(skfmm.distance(self._grid.get_raw_data(), dx=self._grid.get_cell_size()))


    def init_sdf(self, workspace_aabb, cell_size=0.02):
        """
            Initialize this signed distance field.
            @param workspace_aabb - bounding box of the workspace in form of [min_x, min_y, min_z, max_x, max_y, max_z]
            @param cell_size - cell size of the voxel grid (in meters)
        """
        self._grid = VoxelGrid(workspace_aabb, cell_size=cell_size)
        # First compute binary collision map
        start_time = time.time()
        self._compute_bcm()
        print ('Computation of collision binary map took %f s' % (time.time() - start_time))
        # next compute sdf
        start_time = time.time()
        self._compute_sdf()
        print ('Computation of sdf took %f s' % (time.time() - start_time))

    def visualize(self, safe_distance=None):
        """
            Visualizes this sdf in the underlying openrave environment.
            @param safe_distance (optional) - if provided, the visualization colors cells that are more than
                    safe_distance away from any obstacle in the same way as obstacles that are infinitely far away.
        """
        if not self._or_visualization:
            self._or_visualization = ORVoxelGridVisualization(self._env, self._grid)
            self._or_visualization.update(max_sat_value=safe_distance)
        else:
            self._or_visualization.update(max_sat_value=safe_distance)
