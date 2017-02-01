import numpy as np
import transformations
import math
from utils import vec_angel_diff, dist_in_range

# TODO this should be specified in a configuration file
LAST_FINGER_JOINT = 'JF20'

# TODO this should be defined in a super module
class InvalidTriangleException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class RobotiqHand:
    def __init__(self, env=None, hand_file=None):
        self._or_env = env
        self._or_env.Load(hand_file)
        self._or_hand = self._or_env.GetRobots()[0]
        self._plot_handler = []
        self._hand_mani = RobotiqHandVirtualManifold(self._or_hand)
    
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
        frame_hand = self.get_tri_frame(hand_points)
        frame_obj = self.get_tri_frame(obj_points)
        T = transformations.identity_matrix()
        T[0:3, 0:3] = self.get_rotation_matrix(frame_hand, frame_obj)
        new_hand_points = np.transpose(np.dot(T[0:3, 0:3], np.transpose(hand_points)))
        obj_c = np.sum(obj_points, axis=0) / 3.
        new_hand_c = np.sum(new_hand_points, axis=0) / 3.
        
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
        frame = [x, y, z]
        return np.asarray(frame)
    
    def get_rotation_matrix(self, frame1, frame2):
        R = np.dot(np.transpose(frame2), np.linalg.inv(np.transpose(frame1)))
        return R

    """
        Opens the hand until there is no collision anymore.
        @param n_step - maximum number of sampling steps
        @return True if successful, False otherwise
    """
    def avoid_collision_at_fingers(self, n_step):
        if n_step <= 0:
            n_step = 1
        finger_joint_idx = self._or_hand.GetJoint(LAST_FINGER_JOINT).GetDOFIndex()
        start_value = self._or_hand.GetDOFValues()[finger_joint_idx]  # Last joint value opens the fingers
        step = (self._or_hand.GetDOFLimits()[0][finger_joint_idx] - start_value) / n_step
        for i in range(n_step):
            if not self._or_env.CheckCollision(self._or_hand):
                return True
            self._or_hand.SetDOFValues([start_value + i * step], [finger_joint_idx])
        return False

    
class RobotiqHandVirtualManifold:
    """
        Mimic the hand manifold interface from our ICRA'16 paper,
        it is not needed to model a reachability manifold for the Robotiq-S.
    """
    def __init__(self, or_hand, com_center_weight=0.1, pos_reach_weight=10.0, angle_reach_weight=10.0):
        self._or_hand = or_hand
        self._com_center_weight = com_center_weight
        self._pos_reach_weight = pos_reach_weight
        self._angle_reach_weight = angle_reach_weight
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

    def set_parameters(self, com_center_weight=None, pos_reach_weight=None, angle_reach_weight=None):
        if com_center_weight is not None:
            self._com_center_weight = com_center_weight
        if pos_reach_weight is not None:
            self._pos_reach_weight = pos_reach_weight
        if angle_reach_weight is not None:
            self._angle_reach_weight = angle_reach_weight

    def predict_hand_conf(self, q):
        """
            Predict a hand configuration for the encoded grasp q.
            :param q - encoded grasp, see encode_grasp for details.
            :return tuple (res, config), where res is a floating point number
                indicating the reachability of the grasp (the larger, the worse)
                and config a hand configuration that achieves the grasp, if it is feasible,
                else config is a configuration at joint limits.
        """
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
        # TODO need to be tested
        contacts = grasp[:, :3]
        # Let's express the contacts in a frame centered at the center of mass
        center_shift = contacts - obj_com
        # We would like contacts to be close around the center of mass.
        # To measure this, we take the Frobenius norm of center_shift
        d = np.linalg.norm(center_shift)
        vec_10 = grasp[0, :3] - grasp[1, :3]
        center_01 = (grasp[0, :3] + grasp[1, :3]) / 2.
        vec_c2 = grasp[2, :3] - center_01
        dist_10 = np.linalg.norm(vec_10)
        dist_c2 = np.linalg.norm(vec_c2)
        # We want contacts to be centered around the center of mass
        # and at the same time would like to spread the contacts apart, so that
        # we have a high resistance against external torques.
        return dist_10 + dist_c2 - self._com_center_weight * d
        
    def get_pred_res(self, q):
        pos_residual0 = dist_in_range(q[0], self._distance_range_0)
        pos_residual1 = dist_in_range(q[1], self._distance_range_1)
        angle_residual0 = q[2]
        angle_residual1 = q[3]
        r = (pos_residual0 + pos_residual1) * self._pos_reach_weight +\
            self._angle_reach_weight * (angle_residual0 + angle_residual1)
        return r
        
    def encode_grasp(self, grasp):
        """
            Encodes the given grasp (rotationally invariant).
        :param grasp: The grasp to encode. It is assumed the grasp is a matrix of the following format:
            [[position, normal], [position, normal], [position, normal]] where all vectors are defined in
            the object's frame.
        :return:
            A grasp encoding: [distance_01, distance_2c, angle_difference_01, angle_difference_201] where
            distance_01 is the distance between the contact for finger 0 and 1,
            distance_2c is the distance between contact 2 and the center between contact 0 and 1,
            angle_difference_01 is the difference in angle between normals of contact 0 and 1,
            angle_difference_201 is the difference in angle between normals of contact 2 and - avg_normal of
                contact 0 and 1
        """
        vec_10 = grasp[0, :3] - grasp[1, :3]
        center_01 = (grasp[0, :3] + grasp[1, :3]) / 2.
        vec_c2 = grasp[2, :3] - center_01
        distance_10 = np.linalg.norm(vec_10)
        distance_c2 = np.linalg.norm(vec_c2)
        angle_diff_01 = vec_angel_diff(grasp[0, 3:], grasp[1, 3:])
        avg_normal_01 = (grasp[0, 3:] + grasp[1, 3:]) / 2.
        angle_diff_201 = vec_angel_diff(grasp[2, 3:], -avg_normal_01)
        return [distance_10, distance_c2, angle_diff_01, angle_diff_201]
