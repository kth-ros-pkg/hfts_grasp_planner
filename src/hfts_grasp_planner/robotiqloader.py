import numpy as np
import transformations
import math
import os
import logging
from scipy.spatial import KDTree
from utils import vec_angel_diff, dist_in_range

# TODO this should be specified in a configuration file
LAST_FINGER_JOINT = 'finger_2_joint_1'


# TODO this should be defined in a super module
class InvalidTriangleException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class RobotiqHand:
    def __init__(self, env, hand_cache_file, hand_file):
        self._or_env = env
        self._or_env.Load(hand_file)
        self._or_hand = self._or_env.GetRobots()[0]
        self._plot_handler = []
        # self._hand_mani = RobotiqHandVirtualManifold(self._or_hand)
        self._hand_mani = RobotiqHandKDTreeManifold(self, hand_cache_file)

    def __getattr__(self, attr): # composition
        return getattr(self._or_hand, attr)
        
    def get_hand_manifold(self):
        return self._hand_mani
    
    def plot_fingertip_contacts(self):
        self._plot_handler = []
        colors = [np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, 0, 1))]
        tip_link_ids = self.get_fingertip_links()
        point_size = 0.005
        
        for i in range(len(tip_link_ids)):
            link = self._or_hand.GetLink(tip_link_ids[i])
            T = link.GetGlobalMassFrame()
            local_frame_rot = transformations.rotation_matrix(math.pi / 6., [0, 0, 1])[:3, :3]
            T[:3, :3] = T[:3, :3].dot(local_frame_rot)
            
            offset = T[0:3,0:3].dot(self.get_tip_offsets())
            T[0:3,3] = T[0:3,3] + offset
            
            position = T[:3, -1]
            self._plot_handler.append(self._or_env.plot3(points=position, pointsize=point_size, colors=colors[i], drawstyle=1))
            for j in range(3):
                normal = T[:3, j]
                self._plot_handler.append(self._or_env.drawarrow(p1=position, p2=position + 0.05 * normal, linewidth=0.001, color=colors[j]))
    
    def get_tip_offsets(self):
        return np.array([0.025, 0.006, 0.0])

    def get_tip_transforms(self):
        tip_link_ids = self.get_fingertip_links()
        ret = []
        for i in range(len(tip_link_ids)):
            link = self._or_hand.GetLink(tip_link_ids[i])
            T = link.GetGlobalMassFrame()
            local_frame_rot = transformations.rotation_matrix(math.pi / 6., [0, 0, 1])[:3, :3]
            T[:3, :3] = T[:3, :3].dot(local_frame_rot)
            offset = T[0:3,0:3].dot(self.get_tip_offsets())
            T[0:3,3] = T[0:3,3] + offset
            ret.append(T)
        return ret
        
    def get_fingertip_links(self):
        return ['finger_1_link_3', 'finger_2_link_3', 'finger_middle_link_3']
    
    def get_non_fingertip_links(self):
        return ['palm', 'finger_1_link_0', 'finger_1_link_2',
                'finger_2_link_0', 'finger_2_link_1', 'finger_2_link_2',
                'finger_middle_link_0', 'finger_middle_link_1', 'finger_middle_link_2']
    
    def get_tip_pn(self):
        ret = []
        tfs = self.get_tip_transforms()
        for t in tfs:
            ret.append(np.concatenate((t[:3, 3], t[:3, 1])))
        return np.asarray(ret)
    
    def get_ori_tip_pn(self, hand_conf):
        self._or_hand.SetTransform(np.identity(4))
        self._or_hand.SetDOFValues(hand_conf)
        return self.get_tip_pn()
    
    def set_random_conf(self):
        self._lower_limits, self._upper_limits = self._or_hand.GetDOFLimits()
        self._upper_limits[1] = 0.93124747
        self_collision = True
        
        while self_collision:
            ret = []
            for i in range(2):
                ret.append(np.random.uniform(self._lower_limits[i], self._upper_limits[i]))
            self.SetDOFValues(ret)
            self_collision = self._or_hand.CheckSelfCollision()
            
    def get_contact_number(self):
        return 3
        
    def hand_obj_transform(self, hand_points, obj_points):
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
        ori = np.sum(points, axis=0) / 3.
        x = (points[0, :] - ori) / np.linalg.norm(points[0, :] - ori)
        e01 = points[1, :] - points[0, :]
        e02 = points[2, :] - points[0, :]
        e12 = points[2, :] - points[1, :]
        if np.linalg.norm(e01) == 0.0 or np.linalg.norm(e02) == 0.0 or np.linalg.norm(e12) == 0.0:
            raise InvalidTriangleException('Two points are identical')
        z = (np.cross(e02, e01)) / np.linalg.norm(np.cross(e02, e01))
        y = np.cross(z, x)
        frame = np.transpose([x, y, z])
        return np.asarray(frame)

    def comply_fingertips(self, n_step=100):
        """
            Opens and closes the hand until all and only fingertips are in contact.
        :param n_step: maximal number of iterations
        :return:
        """
        joint_index = self.GetJoint(LAST_FINGER_JOINT).GetDOFIndex()
        limit_value = self.GetDOFLimits()[1][joint_index]
        n_step /= 2
        open_succes = self.avoid_collision_at_fingers(n_step)
        if not open_succes:
            return False, False
        curr_conf = self.GetDOFValues()
        step = (limit_value - curr_conf[joint_index]) / n_step
        for i in range(n_step):
            curr_conf[joint_index] += step
            self.SetDOFValues(curr_conf)
            if self.are_fingertips_in_contact():
                return open_succes, True
        return open_succes, False

    def are_fingertips_in_contact(self):
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
        finger_joint_idx = self.GetJoint(LAST_FINGER_JOINT).GetDOFIndex()
        start_value = self.GetDOFValues()[finger_joint_idx]  # Last joint value opens the fingers
        step = (self.GetDOFLimits()[0][finger_joint_idx] - start_value) / n_step
        for i in range(n_step):
            if not self._or_env.CheckCollision(self._or_hand):
                return True
            self.SetDOFValues([start_value + i * step], [finger_joint_idx])
        return False


# TODO should be generalized to any type of hand
class RobotiqHandKDTreeManifold:
    """
        KD tree based hand manifold for the Robotiq hand
    """
    CODE_DIMENSION = 6
    NUM_SAMPLES = 10000

    def __init__(self, or_robot, cache_file_name):
        self._or_robot = or_robot
        self._cache_file_name = cache_file_name
        self._codes = None
        self._hand_configurations = None
        self._kd_tree = None
        self._code_position_scale = 10.0
        self._com_center_weight = 1.0

    def set_parameters(self, com_center_weight=None):
        if com_center_weight is not None:
            self._com_center_weight = com_center_weight

    def load(self):
        if os.path.exists(self._cache_file_name):
            logging.info('[RobotiqHandKDTreeManifold::load] Loading sample data set form disk.')
            data = np.load(self._cache_file_name)
            self._codes = data[:, :self.CODE_DIMENSION]
            self._hand_configurations = data[:, self.CODE_DIMENSION:]
        else:
            logging.info('[RobotiqHandKDTreeManifold::load] No data set available. Generating new...')
            self._sample_configuration_space()
            data = np.concatenate((self._codes, self._hand_configurations), axis=1)
            np.save(self._cache_file_name, data)
        self._kd_tree = KDTree(self._codes)
        # self.test_manifold()

    def _sample_configuration_space(self):
        lower_limits, upper_limits = self._or_robot.GetDOFLimits()
        #TODO can this be done in a niceer way? closing the hand all the way does not make sense
        # TODO hence this limit instead
        upper_limits[1] = 0.93124747
        joint_ranges = np.array(upper_limits) - np.array(lower_limits)
        interpolation_steps = int(math.sqrt(self.NUM_SAMPLES))
        step_sizes = joint_ranges / interpolation_steps
        config = np.array(lower_limits)
        self._hand_configurations = np.zeros((self.NUM_SAMPLES, self._or_robot.GetDOF()))
        self._codes = np.zeros((self.NUM_SAMPLES, self.CODE_DIMENSION))
        sample_idx = 0
        logging.info('[RobotiqHandKDTreeManifold::Sampling %i hand configurations.' % self.NUM_SAMPLES)
        for j0 in range(interpolation_steps):
            config[0] = j0 * step_sizes[0] + lower_limits[0]
            for j1 in range(interpolation_steps):
                config[1] = j1 * step_sizes[1] + lower_limits[1]
                self._or_robot.SetDOFValues(config)
                if self._or_robot.CheckSelfCollision():
                    continue
                fingertip_contacts = self._or_robot.get_ori_tip_pn(config)
                handles = []
                self.draw_contacts(fingertip_contacts, handles)
                self._hand_configurations[sample_idx] = np.array(config)
                self._codes[sample_idx] = self.encode_grasp(fingertip_contacts)
                sample_idx += 1

        self._hand_configurations = self._hand_configurations[:sample_idx, :]
        self._codes = self._codes[:sample_idx, :]
        # TODO see whether we wanna normalize codes
        logging.info('[RobotiqHandKDTreeManifold::Sampling finished. Found %i collision-free hand configurations.' % sample_idx)

    def test_manifold(self):
        """
            For debugging...
            Essentially repeats the sampling procedure, but instead of filling the internal
            database it queries it and compares how accurate the retrieval is.
        """
        lower_limits, upper_limits = self._or_robot.GetDOFLimits()
        upper_limits[1] = 0.93124747
        joint_ranges = np.array(upper_limits) - np.array(lower_limits)
        interpolation_steps = int(math.sqrt(self.NUM_SAMPLES))
        step_sizes = joint_ranges / interpolation_steps
        config = np.array(lower_limits)
        avg_error, min_error, max_error = 0.0, float('inf'), -float('inf')
        num_evaluations = 0

        for j0 in range(interpolation_steps):
            config[0] = j0 * step_sizes[0] + lower_limits[0]
            for j1 in range(interpolation_steps):
                config[1] = j1 * step_sizes[1] + lower_limits[1]
                self._or_robot.SetDOFValues(config)
                if self._or_robot.CheckSelfCollision():
                    continue
                fingertip_contacts = self._or_robot.get_ori_tip_pn(config)
                code = self.encode_grasp(fingertip_contacts)
                distance, retrieved_config = self.predict_hand_conf(code)
                error = np.linalg.norm(retrieved_config - config)
                avg_error += error
                min_error = min(error, min_error)
                max_error = max(error, max_error)
                num_evaluations += 1

        avg_error = avg_error / num_evaluations
        logging.info('[RobotiqHandKDTreeManifold::test_manifold] Average error: %f, max: %f, min: %f' %(avg_error, max_error, min_error))

    def encode_grasp(self, grasp):
        """
            Encodes the given grasp (rotationally invariant).
        """
        code_0 = self.encode_contact_pair(grasp[0], grasp[1])
        code_1 = self.encode_contact_pair(grasp[0], grasp[2])
        code_2 = self.encode_contact_pair(grasp[1], grasp[2])

        # TODO see whether we wanna normalize codes
        return np.concatenate((code_0, code_1, code_2))

    def encode_contact_pair(self, contact_0, contact_1):
        position_diff = np.linalg.norm(contact_0[:3] - contact_1[:3])
        normal_diff = np.linalg.norm(contact_0[3:] - contact_1[3:])
        return np.array([position_diff * self._code_position_scale, normal_diff])

    def predict_hand_conf(self, code):
        distance, index = self._kd_tree.query(code)
        hand_conf = self._hand_configurations[index]
        return distance, hand_conf

    def compute_grasp_quality(self, obj_com, grasp):
        """
        Computes a grasp quality for the given grasp.
        :param obj_com: The center of mass of the object.
        :param grasp: The grasp as matrix
            [[position, normal],
             [position, normal],
             [position, normal]] where all vectors are defined in the object's frame.
        :return: a floating point number representing the quality (the larger, the better)
        """
        # TODO we could use Canny instead
        # The idea is that a grasp with the Robotiq hand is most stable if the contacts
        # span a large triangle and the center of mass of the object is close this triangle's
        # center
        vec_01 = grasp[1, :3] - grasp[0, :3]
        vec_02 = grasp[2, :3] - grasp[0, :3]
        triangle_normal = np.cross(vec_01, vec_02)
        triangle_area = np.linalg.norm(triangle_normal)
        triangle_center = np.sum(grasp, 0)[:3] / 3.0
        return triangle_area - self._com_center_weight * np.linalg.norm(obj_com - triangle_center)

    def draw_contacts(self, poses, handles):
        # TODO this is hard coded for three contacts
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        width = 0.001
        length = 0.05
        env = self._or_robot.GetEnv()
        # Draw planned contacts
        for i in range(poses.shape[0]):
            handles.append(env.drawarrow(poses[i, :3],
                                         poses[i, :3] - length * poses[i, 3:],
                                         width, colors[i]))


class RobotiqHandVirtualManifold:
    """
        Mimic the hand manifold interface from our ICRA'16 paper,
        it is not needed to model a reachability manifold for the Robotiq-S.
    """
    def __init__(self, or_hand, com_center_weight=0.5, pos_reach_weight=5.0, f01_parallelism_weight=1.0,
                 grasp_symmetry_weight=1.0, grasp_flatness_weight=1.0, f2_centralism_weight=1.0):
        self._or_hand = or_hand
        self._com_center_weight = com_center_weight
        self._pos_reach_weight = pos_reach_weight
        self._f01_parallelism_weight = f01_parallelism_weight
        self._grasp_symmetry_weight = grasp_symmetry_weight
        self._grasp_flatness_weight = grasp_flatness_weight
        self._f2_centralism_weight = f2_centralism_weight
        # The distances between fingertip 0 and 1, we can achieve:
        self._distance_range_0 = np.array([0.0255, 0.122])
        # The distances between the center of contacts 0,1 and contact 2, we can achieve:
        self._distance_range_1 = np.array([0, 0.165])
        self._lower_limits, self._upper_limits = self._or_hand.GetDOFLimits()
        # The hand can close into itself, which is not useful for fingertip grasping,
        # so we change the upper limit here:
        self._upper_limits[1] = 0.93124747
        # We use a linear approximation to map desired contact distances to joint angles
        self._lin_factor_0 = (self._upper_limits[0] - self._lower_limits[0]) / \
                             (self._distance_range_0[1] - self._distance_range_0[0])  # for joint 0
        self._lin_factor_1 = (self._upper_limits[1] - self._lower_limits[1]) / \
                             (self._distance_range_1[1] - self._distance_range_1[0])  # for joint 1

    def set_parameters(self, com_center_weight=None, pos_reach_weight=None, f01_parallelism_weight=None,
                       grasp_symmetry_weight=None, grasp_flatness_weight=None, f2_centralism_weight=None):
        if com_center_weight is not None:
            self._com_center_weight = com_center_weight
        if pos_reach_weight is not None:
            self._pos_reach_weight = pos_reach_weight
        if f01_parallelism_weight is not None:
            self._f01_parallelism_weight = f01_parallelism_weight
        if grasp_symmetry_weight is not None:
            self._grasp_symmetry_weight = grasp_symmetry_weight
        if grasp_flatness_weight is not None:
            self._grasp_flatness_weight = grasp_flatness_weight
        if f2_centralism_weight is not None:
            self._f2_centralism_weight = f2_centralism_weight

    def predict_hand_conf(self, q):
        """
            Predict a hand configuration for the encoded grasp q.
            :param q - encoded grasp, see encode_grasp for details.
            :return tuple (res, config), where res is a floating point number
                indicating the reachability of the grasp (the larger, the worse)
                and config a hand configuration that achieves the grasp, if it is feasible,
                else config is a configuration at joint limits.
        """
        if q is None:
            return float('inf'), None
        pos_residual0 = dist_in_range(q[0], self._distance_range_0)
        pos_residual1 = dist_in_range(q[1], self._distance_range_1)
        # Check whether the desired contact distance is within the reachable range
        if pos_residual0 == 0:
            # If so, use the linear approximation to compute a joint value
            joint0 = self._lower_limits[0] + (self._distance_range_0[1] - q[0]) * self._lin_factor_0
        elif pos_residual0 > 0 and q[0] < self._distance_range_0[0]:
            # else, either go the self._upper_limits joint limit
            joint0 = self._upper_limits[0]
        elif pos_residual0 > 0 and q[0] > self._distance_range_0[1]:
            # or the self._lower_limits joint limit depending on whether the desired distance is too small or large
            joint0 = self._lower_limits[0]
        else:
            raise ValueError('[RobotiqHandVirtualManifold::predictHandConf] grasp encoding is incorrect')

        # Do the same for the other joint
        if pos_residual1 == 0:
            joint1 = self._lower_limits[1] + (self._distance_range_1[1] - q[1]) * self._lin_factor_1
        elif pos_residual1 > 0 and q[1] < self._distance_range_1[0]:
            joint1 = self._upper_limits[1]
        elif pos_residual1 > 0 and q[1] > self._distance_range_1[1]:
            joint1 = self._lower_limits[1]
        else:
            raise ValueError('[RobotiqHandVirtualMainfold::predictHandConf] grasp encoding is incorrect')
        # Return the configuration and compute the residual of the grasp
        return self.get_pred_res(q), [joint0, joint1]
    
    def compute_grasp_quality(self, obj_com, grasp):
        """
        Computes a grasp quality for the given grasp.
        :param obj_com: The center of mass of the object.
        :param grasp: The grasp as matrix
            [[position, normal],
             [position, normal],
             [position, normal]] where all vectors are defined in the object's frame.
        :return: a floating point number representing the quality (the larger, the better)
        """
        # The idea is that a grasp with the Robotiq hand is most stable if the contacts
        # span a large triangle and the center of mass of the object is close this triangle's
        # center
        vec_01 = grasp[1, :3] - grasp[0, :3]
        vec_02 = grasp[2, :3] - grasp[0, :3]
        triangle_normal = np.cross(vec_01, vec_02)
        triangle_area = np.linalg.norm(triangle_normal)
        triangle_center = np.sum(grasp, 0)[:3] / 3.0
        return triangle_area - self._com_center_weight * np.linalg.norm(obj_com - triangle_center)

        # TODO need to be tested
        # contacts = grasp[:, :3]
        # # Let's express the contacts in a frame centered at the center of mass
        # center_shift = contacts - obj_com
        # # We would like contacts to be close around the center of mass.
        # # To measure this, we take the Frobenius norm of center_shift
        # d = np.linalg.norm(center_shift)
        # vec_10 = grasp[0, :3] - grasp[1, :3]
        # center_01 = (grasp[0, :3] + grasp[1, :3]) / 2.
        # vec_c2 = grasp[2, :3] - center_01
        # dist_10 = np.linalg.norm(vec_10)
        # dist_c2 = np.linalg.norm(vec_c2)
        # # We want contacts to be centered around the center of mass
        # # and at the same time would like to spread the contacts apart, so that
        # # we have a high resistance against external torques.
        # return dist_10 + dist_c2 - self._com_center_weight * d
        
    def get_pred_res(self, q):
        # pos_residual0 = dist_in_range(q[0], self._distance_range_0)
        # pos_residual1 = dist_in_range(q[1], self._distance_range_1)
        pos_residual0 = self.exp_distance_range(q[0], self._distance_range_0)
        pos_residual1 = self.exp_distance_range(q[1], self._distance_range_1)
        r = self._pos_reach_weight * (pos_residual0 + pos_residual1) +\
            self._f01_parallelism_weight * (1.0 - q[2]) + \
            self._grasp_symmetry_weight * (1.0 + q[3]) + \
            self._grasp_flatness_weight * abs(q[4]) + \
            self._f2_centralism_weight * abs(q[5])
        # r = self._f01_parallelism_weight * (1.0 - q[2]) + \
        #     self._grasp_symmetry_weight * (1.0 + q[3]) + \
        #     self._f2_centralism_weight * abs(q[5])
        assert r >= 0.0
        return r

    @staticmethod
    def encode_grasp(grasp):
        """
            Encodes the given grasp (rotationally invariant).
        :param grasp: The grasp to encode. It is assumed the grasp is a matrix of the following format:
            [[position, normal], [position, normal], [position, normal]] where all vectors are defined in
            the object's frame.
        :return:
            A grasp encoding: [distance_01, distance_2c, parallelism_01, parallelism_201,
            parallelism_triangle, centralism] where
            distance_01 is the distance between the contact for finger 0 and 1,
            distance_2c is the distance between contact 2 and the center between contact 0 and 1,
            parallelism_01 is the dot product of the normals of contact 0 and 1,
            parallelism_201 is the dot product of the avg normal of contact 0,1 and the normal of contact 2
            parallelism_triangle is the sum of the dot products of contact normal i and the normal of the
                                 triangle spanned by all contact points
            centralism is a measure of how centralized contact 2 is with respect to contact 0 and 1, where
                                0.0 means perfectly centralized, < 0.0 biased towards contact 0, > 0.0 towards contact 1
        """
        vec_01 = grasp[1, :3] - grasp[0, :3]
        vec_02 = grasp[2, :3] - grasp[0, :3]
        center_01 = (grasp[0, :3] + grasp[1, :3]) / 2.0
        vec_c2 = grasp[2, :3] - center_01
        avg_normal_01 = (grasp[0, 3:] + grasp[1, 3:]) / 2.
        # Features:
        distance_10 = np.linalg.norm(vec_01)
        distance_c2 = np.linalg.norm(vec_c2)
        parallelism_01 = np.dot(grasp[0, 3:], grasp[1, 3:])
        parallelism_201 = np.dot(grasp[2, 3:], avg_normal_01)
        helper_normal = np.cross(grasp[0, 3:], grasp[1, 3:])
        if np.linalg.norm(helper_normal) == 0.0:
            # In this case normals 0 and 1 are in the same plane
            helper_normal = grasp[0, 3:]
        parallelism_triangle = np.dot(helper_normal, grasp[2, 3:])
        centralism = np.dot(vec_01 / distance_10, vec_02) / distance_10 - 0.5
        if math.isnan(parallelism_triangle):
            # This only happens if the contacts do not span a triangle.
            # In this case the 'triangleness' of the grasp is governed by the parallelism of the contacts
            parallelism_triangle = parallelism_01 + parallelism_201
        if math.isnan(centralism):
            # this happens if contact 0 and 1 are identical
            centralism = 0.0
        return [distance_10, distance_c2, parallelism_01, parallelism_201, parallelism_triangle, centralism]
        # angle_diff_01 = vec_angel_diff(grasp[0, 3:], grasp[1, 3:])
        # angle_diff_201 = vec_angel_diff(grasp[2, 3:], -avg_normal_01)
        # return [distance_10, distance_c2, angle_diff_01, angle_diff_201]

    @staticmethod
    def exp_distance_range(dist, distance_range):
        if dist < distance_range[0]:
            return math.exp(-(dist - distance_range[0])) - 1.0
        elif dist > distance_range[1]:
            return math.exp(dist - distance_range[1]) - 1.0
        else:
            return 0.0
