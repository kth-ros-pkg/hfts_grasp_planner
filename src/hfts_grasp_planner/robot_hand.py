import numpy as np
import transformations
import math
import time
import os
import logging
import yaml
from scipy.spatial import KDTree
from utils import vec_angel_diff, dist_in_range, compute_grasp_stability
from openravepy.misc import DrawAxes


class InvalidTriangleException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

FINGERTIP_TRANSFORMS_KEY = 'fingertip_transforms'
POSITION_KEY = 'position'
ORIENTATION_KEY = 'orientation'
NUM_FINGERTIPS_KEY = 'num_fingertips'
FINGERTIP_LINK_NAMES_KEY = 'fingertip_link_names'
OTHER_LINK_NAMES_KEY = 'other_link_names'
OPEN_HAND_DIR_KEY = 'open_hand_dir'
UPPER_LIMIT_KEY = 'upper_limit'
LOWER_LIMIT_KEY = 'lower_limit'
SYMMETRIES_KEY = 'symmetries'
CODE_POSITION_WEIGHT_KEY = 'code_position_weight'
BOUNDING_RADIUS_KEY = 'bounding_radius'
NUMERICAL_EPSILON = 0.000001


class RobotHand(object):
    def __init__(self, env, cache_file, or_hand_file, hand_config_file):
        """
            Creates a new robot hand.
            :param env:  OpenRAVE environment to use
            :param cache_file: a path to a cache file to store/load reachability data from
            :param or_hand_file: a path to an OpenRAVE model of the hand to use
            :param hand_config_file: a path to a YAML file which contains additional hand specific parameters
        """
        self._or_env = env
        self._or_env.Load(or_hand_file)
        self._or_hand = self._or_env.GetRobots()[0]
        self._plot_handles = []
        self._hand_params = self._load_config(hand_config_file)
        self._fingertip_tfs = self._compute_fn_transforms()
        self._limits = self._get_limits()
        self._hand_mani = ReachabilityKDTree(self, cache_file, self._hand_params)
        self._fingertip_symmetries = self._prepare_symmetries()

    def __getattr__(self, attr): # composition, gets called when this class doesn't have attr.
        return getattr(self._or_hand, attr)  # in this case forward it to the OpenRAVE robot

    def _prepare_symmetries(self):
        """
            Prepares a list of tuples (i, j), where each tuple means that it
            makes sense for a given contact tuple to switch contact assignment for fingertip i and j,
            if the current assignment does not work. This encodes how for instance mirrored grasps
            can be reached. The list is given by the user in the hand parameters.
        """
        symmetries = []
        if SYMMETRIES_KEY not in self._hand_params:
            return symmetries
        link_names = self.get_fingertip_links()
        name_to_idx = dict(zip(link_names, range(len(link_names))))
        symmetries = [(name_to_idx[link_1], name_to_idx[link_2]) for (link_1, link_2) in self._hand_params[SYMMETRIES_KEY]]
        return symmetries

    def _load_config(self, config_file_name):
        """
            Loads the given configuration file and returns the stored dictionary
        """
        with open(config_file_name, 'r') as afile:
            # TODO we could do some sanity checking here on whether we have all required parameters
            file_content = yaml.load(afile)
            if not isinstance(file_content, dict):
                raise ValueError('The hand configuration file needs to contain a dictionary mapping parameter names to values')
            if OTHER_LINK_NAMES_KEY not in file_content:
                all_link_names = [link.GetName() for link in self._or_hand.GetLinks()]
                file_content[OTHER_LINK_NAMES_KEY] = [link_name for link_name in all_link_names if link_name not in file_content[FINGERTIP_LINK_NAMES_KEY]]
            file_content[OPEN_HAND_DIR_KEY] = np.array(file_content[OPEN_HAND_DIR_KEY])
            return file_content

    def _get_limits(self):
        """
            Returns the joint limits of the hand that are useful for grasping.
        """
        limits = list(self._or_hand.GetDOFLimits())
        if UPPER_LIMIT_KEY in self._hand_params:
            limits[1] = np.array(self._hand_params[UPPER_LIMIT_KEY])
        if LOWER_LIMIT_KEY in self._hand_params:
            limits[0] = np.array(self._hand_params[LOWER_LIMIT_KEY])
        return limits

    def GetDOFLimits(self):
        """
            Returns the limits for this hand as they are useful for grasping
        """
        return self._limits

    def _compute_fn_transforms(self):
        """
            Computes fingertip transforms w.r.t to link frames from parameters and returns these.
        """
        fingertip_link_names = self._hand_params[FINGERTIP_LINK_NAMES_KEY]
        fingertip_tf_params = self._hand_params[FINGERTIP_TRANSFORMS_KEY]
        num_fn = self._hand_params[NUM_FINGERTIPS_KEY]
        assert len(fingertip_link_names) == num_fn
        assert len(fingertip_tf_params) == num_fn
        fn_transforms = np.zeros((num_fn, 4, 4))
        for fn in range(num_fn):
            link = self._or_hand.GetLink(fingertip_link_names[fn])
            fn_transforms[fn] = transformations.compose_matrix(translate=fingertip_tf_params[fn][POSITION_KEY],
                                                               angles=fingertip_tf_params[fn][ORIENTATION_KEY])
        return fn_transforms

    def get_hand_manifold(self):
        """
            Return the hand manifold that provides reachability information for contacts.
        """
        return self._hand_mani

    def get_bounding_radius(self):
        return self._hand_params[BOUNDING_RADIUS_KEY]

    def plot_fingertip_contacts(self):
        """
            Render the contact points for this hand.
        """
        self._plot_handles = []
        tip_link_ids = self._hand_params[FINGERTIP_LINK_NAMES_KEY]
        for fn in range(self._hand_params[NUM_FINGERTIPS_KEY]):
            link = self._or_hand.GetLink(tip_link_ids[fn])
            link_tf = link.GetGlobalMassFrame()
            fn_pose = np.dot(link_tf, self._fingertip_tfs[fn])
            self._plot_handles.append(DrawAxes(self._or_env, fn_pose, dist=0.03, linewidth=1.0))

    def get_tip_transforms(self):
        """
            Returns global transforms for each fingertip
        """
        tip_link_ids = self._hand_params[FINGERTIP_LINK_NAMES_KEY]
        ret = []
        for fn in range(self._hand_params[NUM_FINGERTIPS_KEY]):
            link = self._or_hand.GetLink(tip_link_ids[fn])
            link_tf = link.GetGlobalMassFrame()
            fn_pose = np.dot(link_tf, self._fingertip_tfs[fn])
            ret.append(fn_pose)
        return ret

    def get_fingertip_links(self):
        """
            Returns a list of fingertip link names.
        """
        return self._hand_params[FINGERTIP_LINK_NAMES_KEY]

    def get_non_fingertip_links(self):
        """
            Returns a list of non-fingertip link names.
        """
        return self._hand_params[OTHER_LINK_NAMES_KEY]

    def get_tip_pn(self):
        """
            Return a matrix of shape (#fn, 6), where each row is the position and normal of a fingertip
        """
        ret = []
        tfs = self.get_tip_transforms()
        for t in tfs:
            ret.append(np.concatenate((t[:3, 3], t[:3, 2])))
        return np.asarray(ret)

    def get_ori_tip_pn(self, hand_conf):
        """
            Return a matrix of shape (#fn, 6), where each row is the position and normal of a fingertip.
            In contrast to get_tip_pn, the robot is set to the origin and hand dofs are set to hand_conf for this
            operation.
        """
        prev_conf = self._or_hand.GetDOFValues()
        prev_tf = self._or_hand.GetTransform()
        self._or_hand.SetTransform(np.identity(4))
        self._or_hand.SetDOFValues(hand_conf)
        result = self.get_tip_pn()
        self._or_hand.SetDOFValues(prev_conf)
        self._or_hand.SetTransform(prev_tf)
        return result

    def get_contact_number(self):
        """
            Returns the number of fingertips of this hand
        """
        return self._hand_params[NUM_FINGERTIPS_KEY]

    def hand_obj_transform(self, hand_points, obj_points):
        """
            Returns a 4x4 matrix representing the transformation from hand frame to object frame such that
            hand_points (defined in hand frame) and obj_points (defined in object frame, i.e. world frame)
            are aligned in world frame.
        """
        # We align the hand with the object by matching a frame at the grasp center
        frame_hand = self.get_tri_frame(hand_points)  # [x; y; z] of this frame in the hand frame
        frame_obj = self.get_tri_frame(obj_points)  # [x; y; z] of this frame in the object frame
        # Let's build a transformation matrix from this
        T = transformations.identity_matrix()
        # frame_hand is a rotation matrix that rotates the hand frame to our helper frame at the grasp center
        T[0:3, 0:3] = np.dot(frame_obj, np.transpose(frame_hand))  # transpose == inverse for rotation matrices
        # rotate the hand points to a frame that is aligned with the object frame, but located at the grasp center
        # we call this frame rotated hand frame
        new_hand_points = np.transpose(np.dot(T[0:3, 0:3], np.transpose(hand_points)))
        # use this to compute the translation from object to hand frame
        obj_c = np.sum(obj_points, axis=0) / 3.  # the position of the grasp center in object frame
        new_hand_c = np.sum(new_hand_points, axis=0) / 3.  # the position of the grasp center in the rotated hand frame
        # Finally, the translation is from origin to obj_c and then from there in the opposite direction of new_hand_c
        T[:3, -1] = np.transpose(obj_c - new_hand_c)
        return T

    def get_tri_frame(self, points):
        """
            Creates a frame for a given triangle. The input argument points is expected to be
            a matrix of shape (3, 3) where each row represents one corner of the triangle.
            The output is a matrix [x, y, z], where each column vector is one axis of the resulting frame.
            The z axis is orhogonal to the triangle, the x axis points from the center of the triangle
            to points[0, :], the y axis is chosen accordingly.
        """
        # TODO this is a special case for 3 fingertips
        assert(self._hand_params[NUM_FINGERTIPS_KEY] == 3)
        ori = np.sum(points, axis=0) / 3.0
        x = points[0, :] - ori
        x = x / np.linalg.norm(x)
        e01 = points[1, :] - points[0, :]
        e02 = points[2, :] - points[0, :]
        e12 = points[2, :] - points[1, :]
        if np.linalg.norm(e01) == 0.0 or np.linalg.norm(e02) == 0.0 or np.linalg.norm(e12) == 0.0:
            raise InvalidTriangleException('Two points are identical')
        z = np.cross(e02, e01)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        frame = np.transpose([x, y, z])
        return frame

    def comply_fingertips(self, n_step=100):
        """
            Opens and closes the hand until all and only fingertips are in contact.
        :param n_step: maximal number of iterations
        :return: (open_success, in_contact)
        """
        # TODO replacing this with a gradient descent of obstacle penetration cost would probably be much better
        # first try to move hand out of collision
        n_step /= 2
        open_succes = self.avoid_collision_at_fingers(n_step)
        if not open_succes:
            return False, False
        # if this worked, close the hand again until all fingers ar in contact
        num_dofs = self._or_hand.GetDOF()
        curr_conf = self.GetDOFValues()
        close_dir = -self._hand_params[OPEN_HAND_DIR_KEY]
        dir_to_max = self._limits[1] - curr_conf
        dir_to_min = self._limits[0] - curr_conf
        # Compute how much we can move along close_dir until we reach a limit in any DOF
        distances_to_max = [dir_to_max[i] / close_dir[i] for i in range(num_dofs) if close_dir[i] > 0.0]
        scale_range_max = min(distances_to_max) if distances_to_max else float('inf')
        distances_to_min = [dir_to_min[i] / close_dir[i] for i in range(num_dofs) if close_dir[i] < 0.0]
        scale_range_min = min(distances_to_min) if distances_to_min else float('inf')
        # although both scale_range_max and scale_range_min should always be >= 0,
        # it can happen due to numerical noise that one is negative if we are at a limit
        scale_range = min(max(scale_range_max, 0), max(scale_range_min, 0))
        assert(scale_range >= 0.0)
        step = scale_range / n_step
        if step < NUMERICAL_EPSILON:  # if we are already at a limit this might become very small
            return open_succes, self.are_fingertips_in_contact()
        for i in range(n_step):
            curr_conf += step * close_dir
            self.SetDOFValues(curr_conf)
            if self.are_fingertips_in_contact():
                return open_succes, True
        return open_succes, False

    def are_fingertips_in_contact(self):
        """
            Returns whether the fingertip links are in collision
        """
        links = self.get_fingertip_links()
        for link in links:
            if not self._or_env.CheckCollision(self.GetLink(link)):
                return False
        return True

    def avoid_collision_at_fingers(self, n_step):
        """
            Opens the hand until there is no collision anymore.
            :param n_step - maximum number of sampling steps
            :return True if successful, False otherwise
        """
        if n_step <= 0:
            n_step = 1
        num_dofs = self._or_hand.GetDOF()
        curr_conf = self.GetDOFValues()
        open_dir = self._hand_params[OPEN_HAND_DIR_KEY]
        dir_to_max = self._limits[1] - curr_conf
        dir_to_min = self._limits[0] - curr_conf
        # Compute how much we can move along close_dir until we reach a limit in any DOF
        distances_to_max = [dir_to_max[i] / open_dir[i] for i in range(num_dofs) if open_dir[i] > 0.0]
        scale_range_max = min(distances_to_max) if distances_to_max else float('inf')
        distances_to_min = [dir_to_min[i] / open_dir[i] for i in range(num_dofs) if open_dir[i] < 0.0]
        scale_range_min = min(distances_to_min) if distances_to_min else float('inf')
        # although both scale_range_max and scale_range_min should always be >= 0,
        # it can happen due to numerical noise that one is negative if we are at a limit
        scale_range = min(max(scale_range_max, 0), max(scale_range_min, 0))
        assert(scale_range >= 0.0)
        step = scale_range / n_step
        if step < NUMERICAL_EPSILON:  # if we are already at a limit this might become very small
            return not self._or_env.CheckCollision(self._or_hand)
        for i in range(n_step):
            if not self._or_env.CheckCollision(self._or_hand):
                return True
            curr_conf += step * open_dir
            self.SetDOFValues(curr_conf)
        return self._or_env.CheckCollision(self._or_hand)

    def get_fingertip_symmetries(self):
        return self._fingertip_symmetries

# TODO maybe define an interface with the methods the grasp planner needs this to have
class ReachabilityKDTree(object):
    """
        KD tree based hand manifold for a robot hand
    """
    def __init__(self, or_robot, cache_file_name, hand_params, num_samples=1000000):
        """
            Creates a new KD-tree based hand manifold for a robot hand.
            :or_robot: - openrave robot, iu.e. the hand
            :cache_file_name: - filename where to save data
            :hand_params: a dictionary of parameters with the keys the top of this file
            :num_samples: number of samples
        """
        self._or_robot = or_robot
        self._cache_file_name = cache_file_name
        self._codes = None
        self._min_code = None
        self._max_code = None
        self._hand_configurations = None
        self._kd_tree = None
        self._hand_params = hand_params
        self._num_samples = num_samples

    def set_parameters(self, **kwargs):
        if 'num_samples' in kwargs:
            self._num_samples = kwargs['num_samples']
        if 'code_position_weight' in kwargs:
            self._hand_params[CODE_POSITION_WEIGHT_KEY] = kwargs['code_position_weight']

    def load(self):
        if os.path.exists(self._cache_file_name):
            logging.info('[ReachabilityKDTree::load] Loading sample data set from disk.')
            stored_data = np.load(self._cache_file_name)
            meta_data = stored_data[0]
            data = stored_data[1]
            code_dimension = 2 * self._hand_params[NUM_FINGERTIPS_KEY]
            self._codes = data[:, : code_dimension]
            self._hand_configurations = data[:, code_dimension:]
            self._min_code = meta_data[0]
            self._max_code = meta_data[1]
        else:
            logging.info('[ReachabilityKDTree::load] No data set available. Generating new...')
            self._sample_and_create_codes()
            data = np.concatenate((self._codes, self._hand_configurations), axis=1)
            meta_data = np.array([self._min_code, self._max_code])
            store_data = np.array([meta_data, data], dtype=object)
            np.save(self._cache_file_name, store_data)
        self._kd_tree = KDTree(self._codes)

    def _sample_and_create_codes(self, b_force_grid=False):
        """
            Samples the configuration space of the hand and creates hand codes.
            :b_force_grid: If True, the samples are taken from a uniform grid in C-space, else
                they are sampled randomly from a uniform distribution if the hand has more than 2 DOFs.
        """
        lower_limits, upper_limits = self._or_robot.GetDOFLimits()
        lower_limits = np.array(lower_limits)
        joint_ranges = np.array(upper_limits) - lower_limits
        num_dofs = self._or_robot.GetDOF()
        actual_num_samples = self._num_samples
        if b_force_grid or num_dofs <= 2:
            samples_per_dof = int(np.power(self._num_samples, 1.0 / num_dofs))
            # the following expressions generate a matrix where each row is a sample placed on a uniform
            # grid in the hand's configuration space
            sample_values = (np.linspace(lower_limits[i], upper_limits[i], samples_per_dof) for i in range(num_dofs))
            actual_num_samples = np.power(samples_per_dof, num_dofs)
            self._hand_configurations = np.array(np.meshgrid(*sample_values)).T.reshape((actual_num_samples, num_dofs))
        else:
            random_data = np.random.rand(self._num_samples, num_dofs)
            self._hand_configurations = np.apply_along_axis(lambda row: lower_limits + row * joint_ranges, 1, random_data)
        self._codes = np.zeros((actual_num_samples, 2 * self._hand_params[NUM_FINGERTIPS_KEY]))
        logging.info('[ReachabilityKDTree::Evaluating %i hand configurations.' % actual_num_samples)
        # now compute codes for all configurations
        with self._or_robot.GetEnv():
            for sample_idx in xrange(self._num_samples):
                sample = self._hand_configurations[sample_idx]
                self._or_robot.SetDOFValues(sample)
                while self._or_robot.CheckSelfCollision():  # overwrite sample, if it is in collision
                    sample = np.random.rand(1, num_dofs)[0] * joint_ranges + lower_limits
                    self._or_robot.SetDOFValues(sample)
                    self._hand_configurations[sample_idx] = sample
                fingertip_contacts = self._or_robot.get_ori_tip_pn(sample)
                handles = []
                self.draw_contacts(fingertip_contacts, handles)
                self._codes[sample_idx] = self._encode_grasp_non_normalized(fingertip_contacts)
        # Normalize code
        self._min_code = np.min(self._codes, axis=0)
        self._max_code = np.max(self._codes, axis=0)
        self._codes = (self._codes - self._min_code) / (self._max_code - self._min_code)
        logging.info('[ReachabilityKDTree::Sampling finished. Found %i collision-free hand configurations.' % sample_idx)

    def _encode_grasp_non_normalized(self, grasp):
        """
            Encodes the given grasp (rotationally invariant) without normalizing
            TODO: Currently only working for three contact grasps
        """
        code_0 = self.encode_contact_pair(grasp[0], grasp[1])
        code_1 = self.encode_contact_pair(grasp[0], grasp[2])
        code_2 = self.encode_contact_pair(grasp[1], grasp[2])
        return np.concatenate((code_0, code_1, code_2))

    def encode_grasp(self, grasp):
        """
            Encodes the given grasp (rotationally invariant).
            TODO: Currently only working for three contact grasps
        """
        code = self._encode_grasp_non_normalized(grasp)
        return (code - self._min_code) / (self._max_code - self._min_code)

    def encode_contact_pair(self, contact_0, contact_1):
        """
            Encodes a pair of contacts.
            @param contact_0 - [x,y,z,nx,ny,nz]
            @param contact_1 - [x,y,z,nx,ny,nz]
            @return code
        """
        position_diff = np.linalg.norm(contact_0[:3] - contact_1[:3])
        normal_diff = np.linalg.norm(contact_0[3:] - contact_1[3:])
        return np.array([position_diff * self._hand_params[CODE_POSITION_WEIGHT_KEY], normal_diff])

    def predict_hand_conf(self, code):
        distance, index = self._kd_tree.query(code)
        hand_conf = self._hand_configurations[index]
        return distance, hand_conf

    def compute_grasp_quality(self, obj_com, grasp, mu=1.0):
        """
        Computes a grasp quality for the given grasp.
        :param obj_com: The center of mass of the object.
        :param grasp: The grasp as matrix
            [[position, normal],
             [position, normal],
             [position, normal]] where all vectors are defined in the object's frame.
        :param mu: friction coefficient
        :return: a floating point number representing the quality (the larger, the better)
        """
        return compute_grasp_stability(grasp, mu, com=obj_com)

    def draw_contacts(self, poses, handles):
        """
            Draws the given contacts
            TODO: currently only works for three contacts
        """
        # TODO this is hard coded for three contacts
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        width = 0.001
        length = 0.05
        env = self._or_robot.GetEnv()
        # Draw planned contacts
        for i in range(poses.shape[0]):
            handles.append(env.drawarrow(poses[i, :3],
                                         poses[i, :3] + length * poses[i, 3:],
                                         width, colors[i]))
