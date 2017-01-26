#! /usr/bin/python

from plyfile import PlyData
import numpy as np
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs.msg
import rospy
from sklearn.cluster import KMeans as KMeans
import math, copy, os, itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import openravepy as orpy


class ObjectFileIO:
    def __init__(self, data_path, object_identifier, var_filter=True):
        self._obj_id = object_identifier
        self._data_path = data_path
        self._hfts_file = self._data_path + '/' + self._obj_id + '/hfts.npy'
        self._hfts_param_file = self._data_path + '/' + self._obj_id + '/hftsParam.npy'
        self._obj_com_file = self._data_path + '/' + self._obj_id + '/objCOM.npy'
        self._hfts = None
        self._hfts_param = None
        self._obj_com = None
        self._var_filter = var_filter
        self._obj_file_ext = None
        
    def filter_points(self, points):
        kdt = KDTree(points[:, :3], leaf_size = 6, metric = 'euclidean')
        rospy.loginfo('Filtering points for constructing HFTS')
        vld_idx = np.ones(points.shape[0], dtype=bool)
        i = 0
        for p in points:
            nb_idx = kdt.query([p[:3]], k=20, return_distance=False)[0]
            nb_points_normals = points[nb_idx, 3:]
            var = np.var(nb_points_normals, axis = 0)
            if max(var) > 0.2:
                vld_idx[i] = False
            i += 1
            
        points = points[vld_idx, :]
        return points
        
    def get_points(self):
        obj_file = self._data_path + '/' + self._obj_id + '/objectModel'
        try:
            print obj_file + '.ply'
            points = read_ply_file(obj_file + '.ply')
            if self._var_filter:
                points = self.filter_points(points)
                self._obj_file_ext = '.ply'
            return points
        except:
            rospy.loginfo('[objectFileIO] No valid \".ply\" file found for the object: ' + self._obj_id)
        # TODO Implement stl support
        # try:
        #     points = readStlFile(obj_file + '.stl')
        #     if self._var_filter:
        #         points = self.filter_points(points)
        #         self._obj_file_ext = '.stl'
        #     return points
        # except:
        #     rospy.loginfo('[objectFileIO] No valid \".stl\" file found for the object: ' + self._obj_id)
        rospy.logwarn('No previous file found in the database, will proceed with raw point cloud instead.')
        return None

    def get_obj_file_extension(self):
        if self._obj_file_ext is not None:
            return self._obj_file_ext
        obj_file = self._data_path + '/' + self._obj_id + '/objectModel'
        try:
            points = os.path.isfile(obj_file + '.ply')
            self._obj_file_ext = '.ply'
        except:
            rospy.loginfo('[objectFileIO] No \".ply\" file found for the object: ' + self._obj_id)

        if self._obj_file_ext is None:
            try:
                points = os.path.isfile(obj_file + '.stl')
                self._obj_file_ext = '.stl'
            except:
                rospy.loginfo('[objectFileIO] No \".stl\" file found for the object: ' + self._obj_id)
        return self._obj_file_ext
        
    def get_hfts(self, force_new=False):
        if self._hfts is None or self._hfts_param is None:
            if os.path.isfile(self._hfts_file) and not force_new:
                self._hfts = np.load(self._hfts_file)
                self._hfts_param = np.load(self._hfts_param_file)
                self._obj_com = np.load(self._obj_com_file)
            else:
                if not force_new:
                    rospy.logwarn('HFTS is not available in the database')
                points = self.get_points()
                hfts_gen = HFTSGenerator(points)
                hfts_gen.run()
                self._hfts = hfts_gen.get_hfts()
                self._hfts_param = hfts_gen.get_hfts_param()
                hfts_gen.save_hfts(hfts_file=self._hfts_file, hfts_param_file=self._hfts_param_file,
                                   com_file=self._obj_com_file)
        return self._hfts, self._hfts_param.astype(int)
    
    def get_obj_com(self):
        if self._obj_com is None:
            points = self.get_points()
            return np.mean(points[:, :3], axis = 0)
        return self._obj_com

    def show_hfts(self, level):
        # TODO This function is only for debugging purpose, will be removed
        if self._hfts is None:
            self.get_hfts()
        if level > len(self._hfts_param) - 1:
            raise ValueError('[objectFileIO::showHFTS] level ' + str(level) + ' does not exist')
        b_factors = []
        for i in range(level + 1):
            b_factors.append(np.arange(self._hfts_param[i]))
        labels = itertools.product(*b_factors)
        hfts_labels = self._hfts[:, 6:7 + level]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for label in labels:
            idx = np.where((hfts_labels == label).all(axis=1))[0]
            cluster_points = self._hfts[idx, :3]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=np.random.rand(3,1), s = 100)
        plt.show()


class HFTSGenerator:
    # 6 dim of positions and normals + labels
    def __init__(self, points):
        self._point_n = points.shape[0]
        self._obj_com = np.mean(points[:, :3], axis = 0)
        self._points = np.c_[np.arange(self._point_n), points]
        self._pos_weight = 200
        self._branch_factor = 4
        self._first_level_factor = 3
        self._level_n = None
        self._hfts = None
        self._hfts_param = None

    def set_position_weight(self, w):
        self._pos_weight = w

    def set_branch_factor(self, b):
        self._branch_factor = b

    def _cal_levels(self):
        self._level_n = int(math.log(self._point_n / self._first_level_factor, self._branch_factor)) - 1

    def _get_partition_labels(self, points, branch_factor):
        points = copy.deepcopy(points)
        if points.shape[0] < branch_factor:
            self._stop = True
            rospy.loginfo('HFTS generation finished')
            return None

        estimator = KMeans(n_clusters = branch_factor)
        points[:, :3] *= self._pos_weight
        estimator.fit(points)
        return estimator.labels_

    def _compute_hfts(self, curr_points, level = 0):
        if level >= self._level_n:
            return
        idx = curr_points[:, 0].astype(int)
        if level == 0:
            b_factor = self._branch_factor * self._first_level_factor
        else:
            b_factor = self._branch_factor
        points_6D = curr_points[:, 1:]
        curr_labels = self._get_partition_labels(points_6D, b_factor)
        if curr_labels is None:
            self._level_n = level - 1
            return
        self._hfts[idx, level] = curr_labels
        for label in range(b_factor):
            l_idx = np.where(curr_labels == label)[0]
            sub_points = curr_points[l_idx, :]
            self._compute_hfts(sub_points, level + 1)

    def run(self):
        if self._hfts is not None:
            return
        rospy.loginfo('Generating HFTS')
        if self._level_n is None:
            self._cal_levels()

        self._hfts = np.empty([self._point_n, self._level_n])
        self._compute_hfts(self._points)
        self._hfts = self._hfts[:, :self._level_n]
        self._hfts_param = np.empty(self._level_n)

        for i in range(self._level_n):
            if i == 0:
                self._hfts_param[i] = self._branch_factor * self._first_level_factor
            else:
                self._hfts_param[i] = self._branch_factor

    def save_hfts(self, hfts_file, hfts_param_file, com_file):
        data = np.c_[self._points[:, 1:], self._hfts]
        np.save(file=hfts_file, arr=data)
        np.save(file=hfts_param_file, arr=self._hfts_param)
        np.save(file=com_file, arr=self._obj_com)

    def get_hfts(self):
        if self._hfts is None:
            self.run()
        return np.c_[self._points[:, 1:], self._hfts]

    def get_hfts_param(self):
        if self._hfts_param is None:
            self.run()
        return self._hfts_param


def clamp(values, min_values, max_values):
    clamped_values = len(values) * [0.0]
    assert len(values) == len(min_values) and len(values) == len(max_values)
    for i in range(len(values)):
        clamped_values[i] = max(min(values[i], max_values[i]), min_values[i])
    return clamped_values


def read_ply_file(file_id):
    plydata = PlyData.read(file_id)
    vertex = plydata['vertex']
    (x, y, z, nx, ny, nz) = (vertex[t] for t in ('x', 'y', 'z', 'nx', 'ny', 'nz'))
    points = zip(x, y, z, nx, ny, nz)
    return np.asarray(points)


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

    def draw_bounding_box(self, abb, color=[0.3, 0.3, 0.3], width=1.0):
        position = abb.pos()
        extents = abb.extents()
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

