import itertools
import stl as stl_mesh_module
import numpy as np
import rospy
import sklearn.neighbors
import sklearn.cluster
import math
import time
from external.plyfile import PlyData


def sample_face(v0, v1, v2, normal, density):
    """ Samples the face of a triangle face uniformly.
        :param v0 - vertex 0 of the triangle
        :param v1 - vertex 1 of the triangle
        :param v2 - vertex 2 of the triangle
        :param normal - normal of the triangle
        :param density - sample density in points/meter
    """
    points = []
    a, b, c = v2 - v0, v2 - v1, v1 - v0
    a_length, b_length, c_length = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    # We want c to be the longest edge, so rotate the triangle through until this is true
    for i in range(2):
        if c_length < b_length:
            v0, v1, v2 = v1, v2, v0
            a, b, c = v2 - v0, v2 - v1, v1 - v0
            a_length, b_length, c_length = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        elif c_length < a_length:
            v0, v1, v2 = v2, v0, v1
            a, b, c = v2 - v0, v2 - v1, v1 - v0
            a_length, b_length, c_length = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    assert c_length >= b_length and c_length >= a_length
    c_normalized = c / c_length
    h = a - np.dot(a, c_normalized) * a
    height = np.linalg.norm(h)
    number_height_slices = int(height * density)

    # In case the triangle is really small, we just add the center of the face
    if number_height_slices == 0:
        points.append(np.concatenate(((v0 + v1 + v2) / 3.0, normal)))

    for j in range(number_height_slices):
        c_shortened = (v1 + float(j) / number_height_slices * b) - (v0 + float(j) / number_height_slices * a)
        slice_width = np.linalg.norm(c_shortened)
        number_points = int(slice_width * density)
        for i in range(number_points):
            # TODO This doesn't reach the edge b.
            # TODO However, since we are interested in using the points for HFTS generation, points on the edge
            # TODO of a face are not particularly interesting to have anyways
            position = v0 + float(j) / number_height_slices * a + float(i) / number_points * c_shortened
            new_point = np.concatenate((position, normal))
            points.append(new_point)
    return points


def create_contact_points_from_stl(file_name, density=300):
    """ Create a contact point list from an stl mesh.
        The mesh is read in and the faces are sampled to acquire a dense point cloud with normals.
        :param str file_name: the filename of the stl file to read
        :param float density: number of samples per meter
        :return a list of contact points (numpy array) of the shape [pos, normal], i.e. (1, 6)
    """
    stl_mesh = stl_mesh_module.Mesh.from_file(file_name, calculate_normals=False)
    contact_points = []
    # yes, len(stl_mesh.points) is the number of faces and NOT vertices
    for face_idx in range(len(stl_mesh.points)):
        normal_length = np.linalg.norm(stl_mesh.normals[face_idx])
        if normal_length == 0.0:
            stl_mesh.update_normals()
            normal_length = np.linalg.norm(stl_mesh.normals[face_idx])
            if normal_length == 0.0:
                raise IOError('[hfts_generation.py::create_contact_points_from_stl] Could not extract valid normals from the given file ' \
                              + str(file_name))
        points_on_face = sample_face(stl_mesh.v0[face_idx], stl_mesh.v1[face_idx], stl_mesh.v2[face_idx],
                                     stl_mesh.normals[face_idx], density=density)
        contact_points.extend(points_on_face)
    return np.array(contact_points)


def create_contact_points_from_ply(file_name, density=300):
    ply_data = PlyData.read(file_name)
    contact_points = []
    faces = ply_data['face']
    vertices = ply_data['vertex']
    for face in faces:
        v0 = np.array([vertices[face[0][0]][t] for t in ['x', 'y', 'z', 'nx', 'ny', 'nz']])
        v1 = np.array([vertices[face[0][1]][t] for t in ['x', 'y', 'z', 'nx', 'ny', 'nz']])
        v2 = np.array([vertices[face[0][2]][t] for t in ['x', 'y', 'z', 'nx', 'ny', 'nz']])
        # TODO we are throwing information away here. We could instead interpolate vertex normals
        face_normal = np.mean([v0[3:6], v1[3:6], v2[3:6]], axis=0)
        points_on_face = sample_face(v0[:3], v1[:3], v2[:3], face_normal, density=density)
        contact_points.extend(points_on_face)
    return np.array(contact_points)


def filter_unsmooth_points(points, radius, max_variance):
    """ Filter contact points in the given list of points based on the normal variance in their
        Euclidean neighborhood.
        :param points - list/array of points to filter
        :param float radius: size of neighborhood to consider
        :param float max_variance: threshold for maximum allowed variance
        :return a subset of points for which the variance of normals is below the given threshold
    """
    start_time = time.time()
    kdt = sklearn.neighbors.KDTree(points[:, :3], leaf_size=6, metric='euclidean')
    vld_idx = np.ones(points.shape[0], dtype=bool)
    for i in range(len(points)):
        point = points[i]
        nbs_indices = kdt.query_radius(point[:3], radius)[0]
        if len(nbs_indices) == 0:
            continue
        nb_points_normals = points[nbs_indices, 3:]
        var = np.var(nb_points_normals, axis=0)
        if max(var) > max_variance:
            vld_idx[i] = False
    points = points[vld_idx, :]
    print 'filtering points took %fs' % (time.time() - start_time)
    return points


def filter_object_part_points(points, part_description, distance_threshold=0.01):
    """ Filters points based on a part description. Points which are further away from a
        the given object part than the given threshold are filter out.
        :param points - array/list of contact points
        :param part_description - set of points describing the object part to keep
        :param distance_threshold - maximal distance a contact point in points is allowed to have to its closest point
            in part_description
        :return filtered subset of points, for which each point is within distance_threshold to a point from
            part_description
    """
    kdt = sklearn.neighbors.KDTree(part_description, metric='euclidean')
    idx = np.ones(points.shape[0], dtype=bool)
    for i in range(len(points)):
        point = points[i]
        neighbors_in_part = kdt.query_radius(point[:3], distance_threshold, count_only=True)
        idx[i] = neighbors_in_part > 0
    return points[idx, :]


def down_sample_points(points, num_points):
    """ Uniformly down samples the set points to the given amount of points.
        Note that the sampling is done in index space, i.e. it is not guaranteed that the
        returned list of points represents a uniform distribution over an object's surface.
        :param points - set of points to down sample
        :param num_points - number of points to return
    """
    if len(points) <= num_points:
        return points
    idx = np.random.choice(range(len(points)), num_points, replace=False)
    return points[idx]


def read_object_part_description(file_name, part_value=1):
    positions = []
    with open(file_name, 'r') as afile:
        line_number = 0
        for line in afile:
            line_separated = line.split(' ')
            if line_number == 0 and len(line_separated) == 3:
                print 'First line seems to contain approach direction. Skipping..'
                continue
            if len(line_separated) != 4:
                raise IOError('The file %s contains lines of an unexpected format. ' % file_name +
                              'Excepted 4 numbers, but found %i instead in line %i' % (len(line_separated), line_number))
            else:
                x, y, z, point_part_value = map(float, line_separated)
                if point_part_value == part_value:
                    positions.append(np.array([x, y, z]))
            line_number += 1
    return np.array(positions)


class HFTSGenerator:
    # 6 dim of positions and normals + labels
    def __init__(self, points, com):
        self._point_n = points.shape[0]
        self._obj_com = com
        self._points = np.c_[np.arange(self._point_n), points]
        self._pos_weight = 10
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
        points = np.array(points)
        if points.shape[0] < branch_factor:
            rospy.loginfo('HFTS generation finished')
            return None
        estimator = sklearn.cluster.KMeans(n_clusters=branch_factor)
        points[:, :3] *= self._pos_weight
        estimator.fit(points)
        return estimator.labels_

    def _compute_hfts(self, curr_points, level=0):
        # TODO This implementation suffers from very unbalanced point clouds.
        # TODO The hierarchy's depth is governed by the number of points in the smallest cluster.
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
        if self._level_n is None:
            self._cal_levels()

        rospy.loginfo('Generating HFTS')
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


def or_render_hfts(or_env_drawer, hfts, hfts_params, level, transform=None, b_normals=False, size=0.0005):
    bfactors = []
    level = max(0, min(len(hfts_params), level))
    for l in range(level + 1):
        bfactors.append(range(int(hfts_params[l])))

    labels = itertools.product(*bfactors)
    hfts_labels = hfts[:, 6:7 + level]
    for label in labels:
        idx = np.where((hfts_labels == label).all(axis=1))[0]
        cluster_points = hfts[idx, :6]
        color = [0, 0, 0, 1]
        color[:3] = np.random.rand(3, 1)
        or_render_points(or_env_drawer, cluster_points, transform, b_normals=b_normals,
                         color=color, size=size)


def or_render_points(or_env_drawer, contact_points, transform=None, b_normals=False, color=None,
                     size=0.01):
    """ Renders the set of points in openrave.
        :param OpenRAVEDrawer or_env_drawer: OpenRAVE environment to use to render data in.
        :param contact_points - contact points to render
        :param transform - (Optional) transform to apply to points before rendering
        :param b_normals - (Optional) If true, renders also normals, else just points
        :param color - (Optional) Color to use to draw the points in
        :param size - (Optional) The size of each point.
        :return A list of handles that need to stay in memory as long as the data is supposed to be visualized.
    """
    if color is None:
        color = [1, 0, 0, 1]
    if transform is None:
        transform = np.eye(4, 4)
    for contact in contact_points:
        position = np.dot(transform, np.concatenate((contact[:3], [1])))[:3]
        or_env_drawer.draw_bounding_box(abb=None, color=color, position=position, extents=[size/2.0, size/2.0, size/2.0])
        if b_normals:
            normal = np.dot(transform[:3, :3], contact[3:])
            arrow_length = 3 * size
            arrow_width = 0.1 * size
            or_env_drawer.draw_arrow(position, normal, length=arrow_length, width=arrow_width, color=color)
