#! /usr/bin/python

import numpy as np
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs.msg
import rospy
from sklearn.cluster import KMeans as KMeans
import math, copy, os, itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from stl import mesh as stl_mesh_module
from abc import ABCMeta, abstractmethod
import openravepy as orpy
import hfts_grasp_planner.external.transformations as transformations
import hfts_grasp_planner.hfts_generation as hfts_generation
from scipy.spatial import ConvexHull

DEFAULT_HFTS_GENERATION_PARAMS = {'max_normal_variance': 0.2,
                                  'min_contact_patch_radius': 0.015,
                                  'contact_density': 300,
                                  'max_num_points': 10000,
                                  'position_weight': 2,
                                  'branching_factor': 4,
                                  'first_level_branching_factor': 3}


class ObjectIO(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_hfts(self, obj_id, force_new=False):
        pass

    @abstractmethod
    def get_openrave_file_name(self, obj_id):
        pass


class ObjectFileIO(ObjectIO):
    def __init__(self, data_path, var_filter=True,
                 hfts_generation_parameters=DEFAULT_HFTS_GENERATION_PARAMS,
                 max_num_points=10000):
        self._data_path = data_path
        self._b_var_filter = var_filter
        self._hfts_generation_params = hfts_generation_parameters
        self._max_num_points = max_num_points
        self._last_obj_id = None
        self._last_hfts = None
        self._last_hfts_param = None
        self._last_obj_com = None

    def get_points(self, obj_id, b_filter=None):
        if b_filter is None:
            b_filter = self._b_var_filter
        obj_file = self._data_path + '/' + obj_id + '/objectModel'
        file_extension = self.get_obj_file_extension(obj_id)
        points = None
        contact_density = extract_hfts_gen_parameter(self._hfts_generation_params, 'contact_density')
        if file_extension == '.ply':
            points = hfts_generation.create_contact_points_from_ply(file_name=obj_file + file_extension,
                                                                    density=contact_density)
        elif file_extension == '.stl':
            points = hfts_generation.create_contact_points_from_stl(file_name=obj_file + file_extension,
                                                                    density=contact_density)
        # TODO read point cloud if there no files stored.
        # rospy.logwarn('No previous file found in the database, will proceed with raw point cloud instead.')

        if points is not None:
            com = np.mean(points[:, :3], axis=0)
            if b_filter:
                patch_size = extract_hfts_gen_parameter(self._hfts_generation_params,
                                                        'min_contact_patch_radius')
                max_variance = extract_hfts_gen_parameter(self._hfts_generation_params,
                                                          'max_normal_variance')
                points = hfts_generation.filter_unsmooth_points(points,
                                                                radius=patch_size,
                                                                max_variance=max_variance)
                max_num_points = extract_hfts_gen_parameter(self._hfts_generation_params, 'max_num_points')
                points = hfts_generation.down_sample_points(points, max_num_points)
        else:
            rospy.logerr('[ObjectFileIO] Failed to load mesh from ' + str(file_extension) +
                         ' file for object ' + obj_id)
            com = None
        return points, com

    def get_obj_file_extension(self, obj_id):
        obj_file = self._data_path + '/' + obj_id + '/objectModel'
        b_is_valid_file = os.path.exists(obj_file + '.ply') and os.path.isfile(obj_file + '.ply')
        if b_is_valid_file:
            return '.ply'
        b_is_valid_file = os.path.exists(obj_file + '.stl') and os.path.isfile(obj_file + '.stl')
        if b_is_valid_file:
            return '.stl'
        rospy.logerr('[ObjectFileIO::get_obj_file_extension] No compatible file found with prefix name ' + obj_file)
        return None

    def get_openrave_file_name(self, obj_id):
        file_extension = self.get_obj_file_extension(obj_id)
        if file_extension is not None:
            return self._data_path + '/' + obj_id + '/' + 'objectModel' + file_extension
        xml_file_name = self._data_path + '/' + obj_id + '/' + obj_id + '.kinbody.xml'
        b_xml_file_exists = os.path.exists(xml_file_name)
        if b_xml_file_exists:
            return xml_file_name
        return None

    def get_hfts(self, obj_id, force_new=False):
        # Check whether we have an HFTS for this object in memory
        if self._last_obj_id != obj_id or force_new:
            # If not, update
            b_success = self._update_hfts(obj_id, force_new)
            if not b_success:
                return None, None, None
        return self._last_hfts, self._last_hfts_param.astype(int), self._last_obj_com

    def _read_hfts(self, obj_id, hfts_file, hfts_param_file, obj_com_file):
        if os.path.exists(hfts_file) and os.path.isfile(hfts_file) \
                and os.path.exists(hfts_param_file) and os.path.isfile(hfts_param_file) \
                and os.path.exists(obj_com_file) and os.path.isfile(obj_com_file):
            self._last_obj_id = obj_id
            self._last_hfts = np.load(hfts_file)
            self._last_hfts_param = np.load(hfts_param_file)
            self._last_obj_com = np.load(obj_com_file)
            return True
        return False

    def set_hfts_generation_parameters(self, params):
        if type(params) is not dict:
            raise TypeError('ObjectFileIO::set_hfts_generation_parameters] Expected a dictionary, received ' + str(type(params)))
        self._hfts_generation_params = params

    def show_hfts(self, level, or_drawer, object_transform=None, b_normals=False):
        """
        Renders the most recently loaded hfts in OpenRAVE.
        :param level: the level of the hfts to show
        :param or_drawer: an instance of an OpenRAVEDrawer used for rendering
        :param object_transform: An optional transform of the object frame.
        :param b_normals: If true, also renders normals of each point
        """
        if self._last_hfts is None:
            rospy.logerr('[ObjectFileIO::show_hfts] Non hfts model loaded.')
            return
        if level > len(self._last_hfts_param) - 1:
            raise ValueError('[objectFileIO::showHFTS] level ' + str(level) + ' does not exist')
        hfts_generation.or_render_hfts(or_drawer, self._last_hfts, self._last_hfts_param,
                                       level, transform=object_transform, b_normals=b_normals)
        # b_factors = []
        # for i in range(level + 1):
        #     b_factors.append(np.arange(self._last_hfts_param[i]))
        # labels = itertools.product(*b_factors)
        # hfts_labels = self._last_hfts[:, 6:7 + level]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for label in labels:
        #     idx = np.where((hfts_labels == label).all(axis=1))[0]
        #     cluster_points = self._last_hfts[idx, :3]
        #     ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=np.random.rand(3,1), s = 100)
        # plt.show()

    def _update_hfts(self, obj_id, force_new=False):
        """ Updates the cached hfts """
        hfts_file = self._data_path + '/' + obj_id + '/hfts.npy'
        hfts_param_file = self._data_path + '/' + obj_id + '/hftsParam.npy'
        obj_com_file = self._data_path + '/' + obj_id + '/objCOM.npy'
        # If it does not need to be regenerated, try to load it from file
        if not force_new:
            b_hfts_read = self._read_hfts(obj_id, hfts_file, hfts_param_file, obj_com_file)
            if b_hfts_read:
                return True
            rospy.logwarn('HFTS is not available in the database')

        # If we reached this point, we have to generate a new HFTS from mesh/point cloud
        points, com = self.get_points(obj_id)
        if points is None:
            rospy.logerr('Could not generate HFTS for object ' + obj_id)
            return False
        # If we have points, generate an hfts
        hfts_gen = hfts_generation.HFTSGenerator(points, com)
        hfts_gen.set_branch_factor(extract_hfts_gen_parameter(self._hfts_generation_params, 'branching_factor'))
        hfts_gen.set_position_weight(extract_hfts_gen_parameter(self._hfts_generation_params, 'position_weight'))
        hfts_gen.run()
        self._last_obj_id = obj_id
        self._last_hfts = hfts_gen.get_hfts()
        self._last_hfts_param = hfts_gen.get_hfts_param()
        self._last_obj_com = com
        hfts_gen.save_hfts(hfts_file=hfts_file, hfts_param_file=hfts_param_file,
                           com_file=obj_com_file)
        return True


def extract_hfts_gen_parameter(param_dict, name):
    if name in param_dict:
        return param_dict[name]
    elif name in DEFAULT_HFTS_GENERATION_PARAMS:
        return DEFAULT_HFTS_GENERATION_PARAMS[name]
    else:
        raise ValueError('[utils::extract_hfts_gen_parameter] Unknown HFTS generation parameter ' + str(name))


def clamp(values, min_values, max_values):
    clamped_values = len(values) * [0.0]
    assert len(values) == len(min_values) and len(values) == len(max_values)
    for i in range(len(values)):
        clamped_values[i] = max(min(values[i], max_values[i]), min_values[i])
    return clamped_values


def read_stl_file(file_id):
    stl_mesh = stl_mesh_module.Mesh.from_file(file_id, calculate_normals=False)
    points = np.zeros((len(stl_mesh.points), 6))
    # Extract points with normals from the mesh surface
    for face_idx in range(len(stl_mesh.points)):
        # For this, we select the center of each face
        points[face_idx, 0:3] = (stl_mesh.v0[face_idx] + stl_mesh.v1[face_idx] + stl_mesh.v2[face_idx]) / 3.0
        normal_length = np.linalg.norm(stl_mesh.normals[face_idx])
        if normal_length == 0.0:
            stl_mesh.update_normals()
            normal_length = np.linalg.norm(stl_mesh.normals[face_idx])
            if normal_length == 0.0:
                raise IOError('[utils.py::read_stl_file] Could not extract valid normals from the given file ' \
                              + str(file_id))
        points[face_idx, 3:6] = stl_mesh.normals[face_idx] / normal_length
    return points


def create_point_cloud(points):
    point_cloud = PointCloud()
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'
    point_cloud.header = header
    for point in points:
        point_cloud.points.append(Point32(point[0], point[1], point[2]))
    return point_cloud
    

def vec_angel_diff(v0, v1):
    # in radians
    assert len(v0) == len(v1)
    l0 = math.sqrt(np.inner(v0, v0))
    l1 = math.sqrt(np.inner(v1, v1))
    if l0 == 0 or l1 == 0:
        return 0
    x = np.dot(v0, v1) / (l0*l1)
    x = min(1.0, max(-1.0, x)) # fixing math precision error
    angel = math.acos(x)
    return angel


def dist_in_range(d, r):
    if d < r[0]:
        return r[0] - d
    elif d > r[1]:
        return d - r[1]
    else:
        return 0.0


def normal_distance(normals_a, normals_b):
    d = 0.0
    for i in range(len(normals_a)):
        d += vec_angel_diff(normals_a[i], normals_b[i])
    return d


def position_distance(pos_values_a, pos_values_b):
    d = 0.0
    for i in range(len(pos_values_a)):
        d += np.linalg.norm(pos_values_a[i] - pos_values_b[i])
    return d


def generate_wrench_cone(contact, normal, mu, center, face_n):
    ref_vec = np.array([0, 0, 1])
    center = np.array(center)
    contact = np.array(contact)
    normal = np.array(normal)
    forces = []
    angle_step = float(2 * math.pi) / face_n

    # create face_n cone edges
    for i in range(face_n):
        angle = angle_step * i
        x = mu * math.cos(angle)
        y = mu * math.sin(angle)
        z = 1
        forces.append([x, y, z])

    forces = np.asarray(forces)
    rot_angle = transformations.angle_between_vectors(ref_vec, normal)
    axis = np.cross(ref_vec, normal)
    # take care of axis aligned normals
    if np.linalg.norm(axis) > 0.01:
        r_mat = transformations.rotation_matrix(rot_angle, axis)[:3, :3]
    else:
        if np.dot(ref_vec, normal) > 0:
            r_mat = np.identity(3, float)
        else:
            r_mat = np.identity(3, float)
            r_mat[1,1] = -1.
            r_mat[2,2] = -1.

    forces = np.dot(r_mat, np.transpose(forces))
    forces = np.transpose(forces)
    # compute wrenches
    wrenches = []
    for i in range(face_n):
        torque = np.cross((contact - center), forces[i])
        wrenches.append(np.append(forces[i], torque))
    wrenches = np.asarray(wrenches)
    return wrenches


def compute_grasp_stability(grasp_contacts, mu, com=None, face_n=8):
    """ Computes Canny's grasp quality metric for the given n contacts.
        :param grasp_contacts - An nx6 matrix where each row is a contact position and normal
        :param mu - friction coefficient
        :param com - center of mass of the grasped object (assumed to be [0,0,0], if None)
        :param face_n - number of wrench cone faces
    """
    if com is None:
        com = [0, 0, 0]
    wrenches = []
    grasp = np.asarray(grasp_contacts)
    # iterate over each contact
    for i in range(len(grasp)):
        wrench_cone = generate_wrench_cone(grasp[i, :3], grasp[i, 3:], mu, com, face_n)
        for wrench in wrench_cone:
            wrenches.append(list(wrench))
    wrenches = np.asarray(wrenches)
    hull = ConvexHull(wrenches, incremental=False, qhull_options='Pp QJ')
    offsets = -hull.equations[:, -1]
    return min(offsets)


class OpenRAVEDrawer:
    def __init__(self, or_env, robot, debug):
        """
            Create a new OpenRAVEDrawer.
            Parameters:
                or_env - OpenRAVE environment
                robot - OpenRAVE robot
                debug - Boolean flag whether to enable tree drawing
        """
        self.or_env = or_env
        self.robot = robot
        self.debug = debug
        self.handles = []
        self._node_ids = {}

    def clear(self):
        self.handles = []
        self._node_ids = {}

    def get_eef_pose(self, config):
        orig_config = self.robot.GetDOFValues()
        self.robot.SetDOFValues(config)
        manip = self.robot.GetActiveManipulator()
        eef_pose = manip.GetEndEffectorTransform()
        self.robot.SetDOFValues(orig_config)
        return eef_pose

    def draw_tree(self, tree, color):
        if not tree.get_id() in self._node_ids:
            self._node_ids[tree.get_id()] = {}
        node_ids = self._node_ids[tree.get_id()]
        with self.or_env:
            for n in tree._nodes:
                if n.get_id() in node_ids:
                    continue
                else:
                    node_ids[n.get_id()] = True
                eef_pose = self.get_eef_pose(n.get_sample_data().get_configuration())
                if n.get_parent_id() == n.get_id():
                    root_aabb = orpy.AABB(eef_pose[0:3, 3], [0.01, 0.01, 0.01])
                    self.handles.append(self.draw_bounding_box(root_aabb, color, 2.0))
                    continue
                parent_node = tree._nodes[n.get_parent_id()]
                eef_pose_parent = self.get_eef_pose(parent_node.get_sample_data().get_configuration())
                points = [x for x in eef_pose[0:3, 3]]
                points.extend([x for x in eef_pose_parent[0:3, 3]])
                # print numpy.linalg.norm(eef_pose[0:3,3] - eef_pose_parent[0:3, 3])
                handle = self.or_env.drawlinelist(points, 2, colors=color)
                self.handles.append(handle)

    def draw_trees(self, forward_tree, backward_trees=[]):
        if not self.debug:
            return
        # logging.debug('Forward tree size is: ' + str(forwardTree.size()))
        self.draw_tree(forward_tree, color=[1, 0, 0])
        for bTree in backward_trees:
            # logging.debug('Backward tree of size: ' + str(bTree.size()))
            self.draw_tree(bTree, color=[0, 0, 1])

    def draw_arrow(self, point, dir, length=0.04, width=0.01, color=None):
        if color is None:
            color = [1, 0, 0, 1]
        self.handles.append(self.or_env.drawarrow(point, point + length * dir, width, color))

    def draw_pose(self, transform_matrix):
        for i in range(3):
            color = [0, 0, 0, 1]
            color[i] = 1
            self.draw_arrow(transform_matrix[:3, 3], transform_matrix[:3, i], color=color)

    def draw_bounding_box(self, abb=None, color=[0.3, 0.3, 0.3], width=1.0, position=None, extents=None):
        '''
        Draws a bounding box.
        :param abb: OpenRAVE style Axis aligned bounding box. If None, arguements position and extents must not be None.
        :param array color: Array representing color as rgb
        :param float width: Width of the lines to draw
        :param position: Specifies the center position of the bounding box. Must not be None if abb is None.
        :param extents: Specifies the extents of the bounding box (1/2 * [width, height, depth]).
                    Must not be None if abb is None.
        :return: Reference to handles list.
        '''
        if abb is not None:
            position = abb.pos()
            extents = abb.extents()
        if position is None or extents is None:
            raise ValueError('Either abb must not be None or position and extents must not be None')
        points = [[position[0] - extents[0], position[1] - extents[1], position[2] - extents[2]],
                  [position[0] - extents[0], position[1] + extents[1], position[2] - extents[2]],
                  [position[0] - extents[0], position[1] + extents[1], position[2] + extents[2]],
                  [position[0] - extents[0], position[1] - extents[1], position[2] + extents[2]],
                  [position[0] + extents[0], position[1] - extents[1], position[2] - extents[2]],
                  [position[0] + extents[0], position[1] + extents[1], position[2] - extents[2]],
                  [position[0] + extents[0], position[1] + extents[1], position[2] + extents[2]],
                  [position[0] + extents[0], position[1] - extents[1], position[2] + extents[2]]]
        # Back face
        edges = []
        edges.extend(points[0])
        edges.extend(points[1])
        edges.extend(points[1])
        edges.extend(points[2])
        edges.extend(points[2])
        edges.extend(points[3])
        edges.extend(points[3])
        edges.extend(points[0])
        # Front face
        edges.extend(points[4])
        edges.extend(points[5])
        edges.extend(points[5])
        edges.extend(points[6])
        edges.extend(points[6])
        edges.extend(points[7])
        edges.extend(points[7])
        edges.extend(points[4])
        # Sides
        edges.extend(points[0])
        edges.extend(points[4])
        edges.extend(points[3])
        edges.extend(points[7])
        edges.extend(points[2])
        edges.extend(points[6])
        edges.extend(points[1])
        edges.extend(points[5])
        self.handles.append(self.or_env.drawlinelist(edges, width, color))

