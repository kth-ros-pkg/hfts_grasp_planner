#! /usr/bin/python

import numpy as np
from math import exp
import openravepy as orpy
import transformations
from robotiqloader import RobotiqHand, InvalidTriangleException
import sys, time, logging, copy
import itertools, random
from utils import ObjectFileIO, clamp
import rospy


class PlanningSceneInterface(object):

    def __init__(self, or_env, robot_name):
        """ Sets scene information for grasp planning that considers the whole robot.
            @param or_env OpenRAVE environment containing the whole planning scene and robot
            @param robot_name Name of the robot on which the hand is attached (for ik computations)
        """
        self._or_env = or_env
        self._robot = or_env.GetRobot(robot_name)
        self._manip = self._robot.GetActiveManipulator()
        self._arm_ik = orpy.databases.inversekinematics.InverseKinematicsModel(self._robot,
                                                                               iktype=orpy.IkParameterization.Type.Transform6D)
        # Make sure we have an ik solver
        if not self._arm_ik.load():
            rospy.loginfo('Not IKFast solution found. Computing new one...')
            self._arm_ik.autogenerate()
        self._object = None

    def set_target_object(self, obj_name):
        self._object = self._or_env.GetKinBody(obj_name)

    # # TODO this is a hack for the kwr (7DoFs -> need to sample 1 DoF)
    # # TODO this should be done in a robot specific class
    # def seven_dof_ik(self, pose, ik_options, max_iterations=2, free_joint_index=0):
    #     sol = None
    #     lower_limits, upper_limits = self._manip.GetRobot().GetDOFLimits()
    #     min_v, max_v = lower_limits[free_joint_index], upper_limits[free_joint_index]
    #     stride = (max_v - min_v) / 2.0
    #     num_steps = 2
    #     for i in range(max_iterations):
    #         for j in range(1, num_steps, 2):
    #             v = min_v + j * stride
    #             self._robot.SetDOFValues([v], [free_joint_index])
    #             sol = self._manip.FindIKSolution(pose, ik_options)
    #             if sol is not None:
    #                 extended_sol = [v]
    #                 extended_sol.extend(sol)
    #                 return extended_sol
    #         num_steps *= 2
    #         stride /= 2.0
    #     return sol

    def check_arm_ik(self, hand_pose_object, grasp_conf, seed, open_hand_offset):
        with self._or_env:
            # compute target pose in world frame
            object_pose = self._object.GetTransform()
            hand_pose_scene = np.dot(object_pose, hand_pose_object)
            # save current state
            dof_values = self._robot.GetDOFValues()
            # if we have a seed set it
            arm_dofs = self._manip.GetArmIndices()
            hand_dofs = self._manip.GetGripperIndices()
            if seed is not None:
                self._robot.SetDOFValues(seed, dofindices=arm_dofs)
            # Compute a pre-grasp hand configuration and set it
            pre_grasp_conf = np.asarray(grasp_conf) - open_hand_offset
            lower_limits, upper_limits = self._robot.GetDOFLimits(hand_dofs)
            pre_grasp_conf = np.asarray(clamp(pre_grasp_conf, lower_limits, upper_limits))
            self._robot.SetDOFValues(pre_grasp_conf, dofindices=hand_dofs)
            # Now find an ik solution for the target pose with the hand in the pre-grasp configuration
            sol = self._manip.FindIKSolution(hand_pose_scene, orpy.IkFilterOptions.CheckEnvCollisions)
            # sol = self.seven_dof_ik(hand_pose_scene, orpy.IkFilterOptions.CheckEnvCollisions)
            # If that didn't work, try to compute a solution the is in collision (may be useful anyways)
            if sol is None:
                # sol = self.seven_dof_ik(hand_pose_scene, orpy.IkFilterOptions.IgnoreCustomFilters)
                sol = self._manip.FindIKSolution(hand_pose_scene, orpy.IkFilterOptions.IgnoreCustomFilters)
                b_sol_col_free = False
            else:
                b_sol_col_free = True
            # Restore original dof values
            self._robot.SetDOFValues(dof_values)
        return b_sol_col_free, sol, pre_grasp_conf


class HFTSSampler:

    def __init__(self, scene_interface=None, verbose=False, num_hops=2, vis=False):
        self._verbose = verbose
        self._sampler_viewer = vis
        self._orEnv = orpy.Environment() # create openrave environment
        self._orEnv.SetDebugLevel(orpy.DebugLevel.Fatal)
        self._orEnv.GetCollisionChecker().SetCollisionOptions(orpy.CollisionOptions.Contacts)
        if vis:
            self._orEnv.SetViewer('qtcoin') # attach viewer (optional)
        self._scene_or_env = None
        self._hand_loaded = False
        self._scene_interface = scene_interface
        self._obj_loaded = False
        self._mu = 2.
        self._alpha = 2.
        self._max_iters = 40
        self._hops = num_hops
        self._ita = 0.001
        self._pre_gasp_conf = None
        self._grasp_contacts = None
        self._arm_conf = None
        self._hand_pose_lab = None
        self._robot = None
        self._obj = None
        self._obj_com = None
        self._data_labeled = None
        self._hand_manifold = None
        self._num_contacts = None
        self._contact_combinations = []
        self._num_levels = 0
        self._branching_factors = []
        self.handles = []
        self.tip_pn_handler = []

    def __del__(self):
        orpy.RaveDestroy()

    def compute_contact_combinations(self):
        while len(self._contact_combinations) < self._num_levels:
            self._contact_combinations.append([])
        for i in range(self._num_levels):
            self._contact_combinations[i] = set(itertools.product(range(self._branching_factors[i]),
                                                                  repeat=self._num_contacts))

    def compute_allowed_contact_combinations(self, depth, label_cache):
        # Now, for this parent get all possible contacts
        allowed_finger_combos = set(self._contact_combinations[depth])
        # Next, we want to filter out contact combinations that are stored in labelCache
        forbidden_finger_combos = set()
        for graspLabel in label_cache:
            finger_combo = tuple([x[-1] for x in graspLabel])
            forbidden_finger_combos.add(finger_combo)
        # Filter them out
        allowed_finger_combos.difference_update(forbidden_finger_combos)
        return list(allowed_finger_combos)

    def check_grasp_validity(self):
        # Check whether the hand is collision free
        if self._robot.CheckSelfCollision():
            return False
        return self.is_grasp_collision_free()

    def check_arm_grasp_validity(self, grasp_conf, grasp_pose, seed, open_hand_offset=0.1):
        if self._scene_interface is None:
            #TODO Think about what we should do in this case (planning with free-floating hand)
            return True, None, None
        object_hfts_pose = self._obj.GetTransform()  # pose in environment used for contact planning
        hand_pose_object = np.dot(np.linalg.inv(object_hfts_pose), grasp_pose)
        collision_free, arm_conf, pre_grasp_conf = \
            self._scene_interface.check_arm_ik(hand_pose_object,
                                               grasp_conf,
                                               seed=seed,
                                               open_hand_offset=open_hand_offset)
        return collision_free, arm_conf, pre_grasp_conf

    def get_or_hand(self):
        return self._robot

    def load_hand(self, hand_file):
        if not self._hand_loaded:
            # TODO make this Robotiq hand independent (external hand loader)
            self._robot = RobotiqHand(env = self._orEnv, handFile = hand_file)
            self._hand_manifold = self._robot.getHandMani()
            self._num_contacts = self._robot.getContactNumber()
            shift = transformations.identity_matrix()
            shift[0, -1] = 0.2
            self._robot.SetTransform(shift)
            rospy.loginfo('Hand loaded in OpenRAVE environment')
            self._hand_loaded = True

    def load_object(self, data_path, obj_id, obj_id_scene=None):
        object_io = ObjectFileIO(data_path, obj_id)
        self._data_labeled, self._branching_factors = object_io.getHFTS()
        self._num_levels = len(self._branching_factors)
        self._obj_loaded = self._orEnv.Load(data_path + '/' + obj_id + '/objectModel' + object_io.getObjFileExtension())
        self._obj = self._orEnv.GetKinBody('objectModel')
        self._obj_com = object_io.getObjCOM()
        rospy.loginfo('Object loaded in OpenRAVE environment')
        if self._scene_interface is not None:
            if obj_id_scene is not None:
                self._scene_interface.set_target_object(obj_id_scene)
            else:
                self._scene_interface.set_target_object(obj_id)
        self.compute_contact_combinations()
        self._obj_loaded = True

    def pick_new_start_node(self):
        num_nodes_top_level = self._branching_factors[0]
        contact_label = []
        for i in range(self._num_contacts):
            contact_label.append([random.choice(range(num_nodes_top_level + 1))])
        return contact_label

    def clear_config_cache(self):
        self._pre_gasp_conf = None
        self._grasp_contacts = None
        self._arm_conf = None
        self._hand_pose_lab = None
        shift = transformations.identity_matrix()
        shift[0,-1] = 0.2
        self._robot.SetTransform(shift)
        # Set hand to default (mean) configuration
        mean_values = map(lambda min_v, max_v: (min_v + max_v) / 2.0,
                          self._robot.GetDOFLimits()[0],
                          self._robot.GetDOFLimits()[1])
        self._robot.SetDOFValues(mean_values, range(len(mean_values)))
        self.handles = []
        self.tip_pn_handler = []

    def sample_grasp(self, node, depth_limit, post_opt=False, label_cache=None, open_hand_offset=0.1):
        if depth_limit < 0:
            raise ValueError('HFTSSampler::sample_grasp depth limit must be greater or equal to zero.')

        if node.get_depth() >= self._num_levels:
            raise ValueError('HFTSSampler::sample_grasp input node has an invalid depth')
            
        if node.get_depth() + depth_limit >= self._num_levels:
            depth_limit = self._num_levels - node.get_depth() - 1 # cap

        seed_ik = None
        if node.get_depth() == 0: # at root
            contact_label = self.pick_new_start_node()
            best_o = -np.inf  # need to also consider non-root nodes
        else:
            # If we are not at a leaf node, go down in the hierarchy
            seed_ik = node.getArmConfig()
            contact_label = copy.deepcopy(node.get_labels())
            best_o, contact_label = self.extend_hfts_node(contact_label)

        allowed_finger_combos = None
        if label_cache is not None:
            # TODO This currently only works for hops == 2
            assert self._hops == 2
            allowed_finger_combos = self.compute_allowed_contact_combinations(node.get_depth(), label_cache)
            rospy.logdebug('[HFTSSampler::sample_grasp] We have %i allowed contacts' % len(allowed_finger_combos))
            if len(allowed_finger_combos) == 0:
                rospy.logwarn('[HFTSSampler::sample_grasp] We have no allowed contacts left! Aborting.')
                return node

        self.clear_config_cache()
        depth_limit -= 1
        rospy.logdebug('[HFTSSampler::sample_grasp] Sampling a grasp; %i number of iterations' % self._max_iters)

        while depth_limit >= 0:
            # do stochastic optimization until depth_limit is reached
            for iter_now in range(self._max_iters):
                labels_tmp = self.get_sibling_labels(curr_labels=contact_label,
                                                     allowed_finger_combos=allowed_finger_combos)
                s_tmp, r_tmp, o_tmp = self.evaluate_grasp(labels_tmp)

                if self.shc_evaluation(o_tmp, best_o):
                    contact_label = labels_tmp
                    best_o = o_tmp

            # descend to next level
            best_o, contact_label = self.extend_hfts_node(contact_label)
            depth_limit -= 1

        # Create output
        grasp_conf, fingertip_poses = self.compose_grasp_info(contact_label)
        if post_opt:
            rospy.logdebug('[HFTSSampler::sample_grasp] Doing post optimization for node %s' % str(contact_label))
        # Compute grasp quality (a combination of stability, reachability and collision conditions)
        try:
            # sample_q, stability = self.simulate_grasp(grasp_conf=grasp_conf, grasp_pose=grasp_pose,
            #                                           post_opt=post_opt)
            b_robotiq_ok = self.simulate_grasp(grasp_conf=grasp_conf, fingertip_poses=fingertip_poses,
                                               post_opt=post_opt)
            if b_robotiq_ok:
                sample_q = 0
                stability = best_o
            else:
                sample_q = 4
                stability = 0.0
        except InvalidTriangleException:
            grasp_conf = None
            sample_q = 4
            stability = 0.0

        is_leaf = (len(contact_label[0]) == self._num_levels)
        is_goal_sample = (sample_q == 0) and is_leaf
        if not is_goal_sample and grasp_conf is not None:
            rospy.logdebug('[HFTSSampler::sample_grasp] Approximate has final quality: %i' % sample_q)
            self.avoid_collision_at_fingers(n_step=20)
            open_hand_offset = 0.0

        logging.debug('[HFTSSampler::sample_grasp] We sampled a grasp on level ' + str(len(contact_label[0])))
        if is_goal_sample:
            logging.debug('[HFTSSampler::sample_grasp] We sampled a goal grasp (might be in collision)!')
        if is_leaf:
            logging.debug('[HFTSSampler::sample_grasp] We sampled a leaf')

        if grasp_conf is not None:
            grasp_pose = self._robot.GetTransform()
            collision_free_arm_ik, arm_conf, pre_grasp_conf =\
                self.check_arm_grasp_validity(grasp_conf=grasp_conf,
                                              grasp_pose=grasp_pose,
                                              seed=seed_ik, open_hand_offset=open_hand_offset)
        else:
            collision_free_arm_ik = False

        depth = len(contact_label[0])
        possible_num_children, possible_num_leaves = self.get_branch_information(depth)
        return HFTSNode(labels=contact_label, hand_conf=grasp_conf,
                        pre_grasp_conf=pre_grasp_conf, arm_conf=arm_conf,
                        is_goal=is_goal_sample, is_leaf=is_leaf, is_valid=collision_free_arm_ik,
                        num_possible_children=possible_num_children, num_possible_leaves=possible_num_leaves,
                        hand_transform= self._robot.GetTransform())

    def simulate_grasp(self, grasp_conf, fingertip_poses, post_opt=False):
        # TODO this method as it is right now is only useful for the Robotiq hand.
        self._robot.SetDOFValues(grasp_conf)
        try:
            T = self._robot.HandObjTransform(fingertip_poses[:3, :3], self._grasp_contacts[:, :3])
            self._robot.SetTransform(T)
        except:
            # TODO figure out what exceptions we expect here and wanna catch
            return False
        self.comply_eef()
        if self.check_grasp_validity():
            return True
        self.swap_contacts([0, 1])
        self._robot.SetDOFValues(grasp_conf)
        try:
            T = self._robot.HandObjTransform(fingertip_poses[:3, :3], self._grasp_contacts[:, :3])
            self._robot.SetTransform(T)
        except:
            return False
        self.comply_eef()
        return self.check_grasp_validity()

    def compose_grasp_info(self, contact_labels):
        contacts = [] # a list of contact positions and normals
        for i in range(self._num_contacts):
            p, n = self.cluster_repr(contact_labels[i])
            contacts.append(list(p) + list(n))
        # TODO make this to a non-object-global variable
        self._grasp_contacts = np.asarray(contacts)
        code_tmp = self._hand_manifold.encodeGrasp(self._grasp_contacts)
        dummy, grasp_conf  = self._hand_manifold.predictHandConf(code_tmp)
        grasp_pose = self._robot.getOriTipPN(grasp_conf)
        return grasp_conf, grasp_pose

    def extend_hfts_node(self, old_labels):
        for label in old_labels:
            label.append(np.random.randint(self._branching_factors[len(label)]))
        s_tmp, r_tmp, o_tmp = self.evaluate_grasp(old_labels)
        return o_tmp, old_labels
 
    def cluster_repr(self, label):
        level = len(label) - 1 # indexed from 0
        idx = np.where((self._data_labeled[:, 6:7 + level] == label).all(axis=1))
        points = [self._data_labeled[t, 0:3] for t in idx][0]
        normals = [self._data_labeled[t, 3:6] for t in idx][0]

        pos = np.sum(points, axis=0) / len(idx[0])
        normal = np.sum(normals, axis=0) / len(idx[0])
        normal /= np.linalg.norm(normal)
        return pos, -normal
 
    def swap_contacts(self, rows):
        frm = rows[0]
        to = rows[1]
        self._grasp_contacts[[frm, to], :] = self._grasp_contacts[[to, frm], :]

    def evaluate_grasp(self, contact_label):
        contacts = [] # a list of contact positions and normals
        for i in range(self._num_contacts):
            p, n = self.cluster_repr(contact_label[i])
            contacts.append(list(p) + list(n))
        contacts = np.asarray(contacts)
        s_tmp = self._hand_manifold.computeGraspQuality(self._obj_com, contacts)
        code_tmp = self._hand_manifold.encodeGrasp(contacts)
        r_tmp, dummy = self._hand_manifold.predictHandConf(code_tmp)
        # o_tmp = s_tmp - self._alpha * r_tmp
        # TODO: Research topic. This is kind of hack. Another objective function might be better
        o_tmp = s_tmp / (r_tmp + 0.000001)
        return s_tmp, r_tmp, o_tmp

    def shc_evaluation(self, o_tmp, best_o):
        if best_o < o_tmp:
            return True
        else:
            return False
        # v = (bestO - o_tmp) / self._ita
        # if v < 0: #python overflow
        #     return True
        # else:
        #     return False
        # p = 1. / (1 + exp(v))
        # return  p > np.random.uniform()

    def get_sibling_label(self, label):
        ret = []
        if len(label) <= self._hops / 2:
            for i in range(len(label)):
                ret.append(np.random.randint(self._branching_factors[i]))
        else:
            match_len = len(label) - self._hops / 2
            ret = label[:match_len]
            for i in range(len(label) - match_len):
                ret.append(np.random.randint(self._branching_factors[i + match_len]))
        return ret

    def get_sibling_labels(self, curr_labels, allowed_finger_combos=None):
        labels_tmp = []
        if allowed_finger_combos is None:
            for i in range(self._num_contacts):
                tmp = []
                # while tmp in labels_tmp or len(tmp) == 0:
                # TODO what is this loop for? we would never get this result from get_sibling_label
                while len(tmp) == 0:
                    tmp = self.get_sibling_label(curr_labels[i])
                labels_tmp.append(tmp)
        else:
            finger_combo = random.choice(allowed_finger_combos)
            for i in range(self._num_contacts):
                tmp = list(curr_labels[i])
                tmp[-1] = finger_combo[i]
                labels_tmp.append(tmp)
        return labels_tmp

    def get_branch_information(self, level):
        if level < self.get_maximum_depth():
            possible_num_children = pow(self._branching_factors[level] + 1, self._num_contacts)
            possible_num_leaves = 1
            for d in range(level, self.get_maximum_depth()):
                possible_num_leaves *= pow(self._branching_factors[level] + 1, self._num_contacts)
        else:
            possible_num_children = 0
            possible_num_leaves = 1
        return possible_num_children, possible_num_leaves

    def get_root_node(self):
        possible_num_children, possible_num_leaves = self.get_branch_information(0)
        return HFTSNode(num_possible_children=possible_num_children,
                        num_possible_leaves=possible_num_leaves)

    def get_maximum_depth(self):
        return self._num_levels

    def set_max_iter(self, m):
        assert m > 0
        self._max_iters = m

    def comply_eef(self):
        # TODO this seems to be Robotiq hand specific
        curr_conf = self._robot.GetDOFValues()
        for i in range(100):
            curr_conf[1] += 0.01
            self._robot.SetDOFValues(curr_conf)
            if self.are_fingertips_in_contact():
                break

    def are_fingertips_in_contact(self):
        links = self._robot.getFingertipLinks()
        for link in links:
            if not self._orEnv.CheckCollision(self._robot.GetLink(link)):
                return False
        return True
    
    def is_grasp_collision_free(self):
        links = self._robot.getNonFingertipLinks()
        for link in links:
            if self._orEnv.CheckCollision(self._robot.GetLink(link)):
                return False
        return True

    def plot_clusters(self, contact_labels):
        if not self._sampler_viewer:
            return
        self.cloud_plot = []
        colors = [np.array((1,0,0)), np.array((0,1,0)), np.array((0,0,1))]
        for i in range(3):
            label = contact_labels[i]
            level = len(label) - 1 # indexed from 0
            idx = np.where((self._data_labeled[:, 6:7 + level] == label).all(axis=1))
            points = [self._data_labeled[t, 0:3] for t in idx][0]
            points = np.asarray(points)
            self.cloud_plot.append(self._orEnv.plot3(points=points, pointsize=0.006, colors=colors[i], drawstyle=1))

    def avoid_collision_at_fingers(self, n_step):
        # TODO auto open each dof individually until fingers are not in collision anymore
        # TODO This is most probably hand specific... Maybe move hand specific code to a different class?
        pass


class HFTSNode:
    def __init__(self, labels=None, hand_conf=None, hand_transform=None,
                 pre_grasp_conf=None, arm_conf=None, is_leaf=False, is_valid=False, is_goal=False,
                 num_possible_children=0, num_possible_leaves=0, quality=0.0):
        # None values represent the root node
        if labels is None:
            self._depth = 0
        else:
            self._depth = len(labels[0])

        self._labels = labels
        self._hand_config = hand_conf
        self._hand_transform = hand_transform
        self._is_goal = is_goal
        self._is_leaf = is_leaf
        self._is_valid = is_valid
        self._pre_grasp_conf = pre_grasp_conf
        self._arm_conf = arm_conf
        self._num_possible_children = num_possible_children
        self._num_possible_leaves = num_possible_leaves
        self._quality = quality
        
    def get_labels(self):
        return self._labels

    def get_depth(self):
        return self._depth

    def get_hand_config(self):
        return self._hand_config
    
    def get_pre_grasp_config(self):
        return self._pre_grasp_conf

    def is_goal(self):
        return self._is_goal
    
    def get_hand_transform(self):
        return self._hand_transform

    def get_arm_configuration(self):
        return self._arm_conf
        
    def get_unique_label(self):
        if self._labels is None:
            return 'root'
        label = []
        for finger_label in self._labels:
            label.extend(finger_label)
        return str(label)
    
    def is_extendible(self):
        return not self._is_leaf
    
    def is_leaf(self):
        return self._is_leaf

    def is_valid(self):
        return self._is_valid

    def get_num_possible_children(self):
        return self._num_possible_children

    def get_num_possible_leaves(self):
        return self._num_possible_leaves
    
    def get_quality(self):
        return self._quality

