#! /usr/bin/python

import numpy as np
import math
from scipy.spatial import KDTree
import openravepy as orpy
import transformations
from robotiqloader import RobotiqHand, InvalidTriangleException
import sys, time, logging, copy
import itertools
from utils import ObjectFileIO, clamp, compute_grasp_stability, normal_distance, position_distance, dist_in_range
import rospy
import scipy.optimize


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
            rospy.loginfo('No IKFast solver found. Generating new one...')
            self._arm_ik.autogenerate()
        self._object = None

    def set_target_object(self, obj_name):
        self._object = self._or_env.GetKinBody(obj_name)

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
            # If that didn't work, try to compute a solution that is in collision (may be useful anyways)
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
    def __init__(self, object_io_interface, scene_interface=None, verbose=False, num_hops=2, vis=False):
        self._verbose = verbose
        self._sampler_viewer = vis
        self._orEnv = orpy.Environment() # create openrave environment
        self._orEnv.SetDebugLevel(orpy.DebugLevel.Fatal)
        self._orEnv.GetCollisionChecker().SetCollisionOptions(orpy.CollisionOptions.Contacts)
        if vis:
            self._orEnv.SetViewer('qtcoin') # attach viewer (optional)
            self._or_handles = []
        else:
            self._or_handles = None
        self._scene_or_env = None
        self._hand_loaded = False
        self._scene_interface = scene_interface
        self._obj_loaded = False
        self._max_iters = 40
        self._reachability_weight = 1.0
        self._mu = 2.0
        self._min_stability = 0.0
        self._b_force_new_hfts = False
        self._object_kd_tree = None
        self._object_points = None
        # self._hops = num_hops
        # TODO remove this aga
        self._hops = 2
        self._robot = None
        self._obj = None
        self._obj_com = None
        self._data_labeled = None
        self._hand_manifold = None
        self._num_contacts = None
        self._contact_combinations = []
        self._num_levels = 0
        self._branching_factors = []
        self._object_io_interface = object_io_interface

    def __del__(self):
        orpy.RaveDestroy()

    def check_arm_grasp_validity(self, grasp_conf, grasp_pose, seed, open_hand_offset=0.1):
        if self._scene_interface is None:
            #TODO Think about what we should do in this case (planning with free-floating hand)
            return True, None, None
        object_hfts_pose = self._obj.GetTransform()  # pose in environment used for contact planning
        hand_pose_object_frame = np.dot(np.linalg.inv(object_hfts_pose), grasp_pose)
        # hand_pose_world = np.dot(object_hfts_pose, grasp_pose)
        collision_free, arm_conf, pre_grasp_conf = \
            self._scene_interface.check_arm_ik(hand_pose_object_frame,
                                               grasp_conf,
                                               seed=seed,
                                               open_hand_offset=open_hand_offset)
        return collision_free, arm_conf, pre_grasp_conf

    def check_grasp_validity(self):
        # Check whether the hand is collision free
        if self._robot.CheckSelfCollision():
            return False
        real_contacts = self.get_real_contacts()
        # self.draw_contacts(real_contacts)
        stability = compute_grasp_stability(grasp_contacts=real_contacts,
                                            mu=self._mu)
        return stability > self._min_stability and self.is_grasp_collision_free()

    def create_object_kd_tree(self, points):
        self._object_kd_tree = KDTree(points[:, :3])
        self._object_points = points

    def compute_allowed_contact_combinations(self, depth, label_cache):
        # Now, for this parent get all possible contacts
        allowed_finger_combos = set(self._contact_combinations[depth])
        # Next, we want to filter out contact combinations that are stored in labelCache
        forbidden_finger_combos = set()
        for grasp_label in label_cache:
            finger_combo = tuple([x[-1] for x in grasp_label])
            forbidden_finger_combos.add(finger_combo)
        # Filter them out
        allowed_finger_combos.difference_update(forbidden_finger_combos)
        return list(allowed_finger_combos)

    def compute_contact_combinations(self):
        while len(self._contact_combinations) < self._num_levels:
            self._contact_combinations.append([])
        for i in range(self._num_levels):
            self._contact_combinations[i] = set(itertools.product(range(self._branching_factors[i]),
                                                                  repeat=self._num_contacts))

    def compose_grasp_info(self, contact_labels):
        contacts = [] # a list of contact positions and normals
        for i in range(self._num_contacts):
            p, n = self.get_cluster_repr(contact_labels[i])
            contacts.append(list(p) + list(n))
        object_contacts = np.asarray(contacts)
        code_tmp = self._hand_manifold.encode_grasp(object_contacts)
        dummy, grasp_conf = self._hand_manifold.predict_hand_conf(code_tmp)
        hand_contacts = self._robot.get_ori_tip_pn(grasp_conf)
        return grasp_conf, object_contacts, hand_contacts

    def _debug_visualize_quality(self, labels, quality, handles):
        grasp_conf, object_contacts, hand_contacts = self.compose_grasp_info(labels)
        self._robot.SetVisible(False)
        handles.append(self._draw_contacts_quality(object_contacts, quality))

    def _draw_contacts_quality(self, object_contacts, quality):
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        quality = min(abs(quality), 0.005)
        width = 0.003
        length = max((1.0 - abs(quality) / 0.005) * 0.05, 0.001)
        # Draw planned contacts
        arrow_handles = []
        for i in range(object_contacts.shape[0]):
            arrow_handles.append(self._orEnv.drawarrow(object_contacts[i, :3],
                                                       object_contacts[i, :3] - length * object_contacts[i, 3:],
                                                       width, colors[i]))
        return arrow_handles

    def _debug_visualize(self, labels, handle_index=-1):
        grasp_conf, object_contacts, hand_contacts = self.compose_grasp_info(labels)
        rospy.logwarn('Debug visualize')
        # self._robot.SetVisible(False)
        # self.draw_contacts(object_contacts, handle_index=handle_index)
        # time.sleep(1.0)
        # self._robot.SetVisible(True)

    def draw_contacts(self, object_contacts, handle_index=-1):
        if len(self._or_handles) == 0:
            self._or_handles.append(None)
            self._or_handles.append(None)
        # TODO this is hard coded for three contacts
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        if handle_index != 0:
            width = 0.003
            length = 0.05
        else:
            width = 0.001
            length = 0.1
        # Draw planned contacts
        arrow_handles = []
        for i in range(object_contacts.shape[0]):
            arrow_handles.append(self._orEnv.drawarrow(object_contacts[i, :3],
                                                       object_contacts[i, :3] - length * object_contacts[i, 3:],
                                                       width, colors[i]))
        self._or_handles[handle_index] = arrow_handles

    def evaluate_grasp(self, contact_label):
        contacts = [] # a list of contact positions and normals
        for i in range(self._num_contacts):
            p, n = self.get_cluster_repr(contact_label[i])
            contacts.append(list(p) + list(n))
        contacts = np.asarray(contacts)
        # self.draw_contacts(contacts)
        s_tmp = self._hand_manifold.compute_grasp_quality(self._obj_com, contacts)
        code_tmp = self._hand_manifold.encode_grasp(contacts)
        r_tmp, dummy = self._hand_manifold.predict_hand_conf(code_tmp)
        # TODO: Research topic. This is kind of hack. Another objective function might be better
        # o_tmp = s_tmp / (r_tmp + 0.000001)
        o_tmp = s_tmp - self._reachability_weight * r_tmp
        assert not math.isnan(o_tmp) and not math.isinf(math.fabs(o_tmp))
        # o_tmp = s_tmp / (r_tmp + 1.0)
        # return s_tmp, r_tmp, o_tmp
        return s_tmp, r_tmp, -r_tmp

    def extend_hfts_node(self, old_labels, allowed_finger_combos=None):
        new_depth = len(old_labels[0])  # a label has length depth + 1
        if allowed_finger_combos is not None:
            fingertip_assignments = np.random.choice(allowed_finger_combos)
        else:
            fingertip_assignments = np.random.choice(self._branching_factors[new_depth],
                                                     self._num_contacts,
                                                     replace=True)

        for label, assignment in itertools.izip(old_labels, fingertip_assignments):
            label.append(assignment)
        s_tmp, r_tmp, o_tmp = self.evaluate_grasp(old_labels)
        # self._debug_visualize(old_labels, 0)
        return o_tmp, old_labels

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

    def get_cluster_repr(self, label):
        level = len(label) - 1 # indexed from 0
        idx = np.where((self._data_labeled[:, 6:7 + level] == label).all(axis=1))
        points = [self._data_labeled[t, 0:3] for t in idx][0]
        normals = [self._data_labeled[t, 3:6] for t in idx][0]

        pos = np.sum(points, axis=0) / len(idx[0])
        normal = np.sum(normals, axis=0) / len(idx[0])
        normal /= np.linalg.norm(normal)
        return pos, -normal

    def get_maximum_depth(self):
        return self._num_levels

    def get_or_hand(self):
        return self._robot

    def get_random_sibling_label(self, label):
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

    def get_random_sibling_labels(self, curr_labels, allowed_finger_combos=None):
        labels_tmp = []
        if allowed_finger_combos is None:
            for i in range(self._num_contacts):
                tmp = self.get_random_sibling_label(curr_labels[i])
                labels_tmp.append(tmp)
        else:
            finger_combo = np.random.choice(allowed_finger_combos)
            for i in range(self._num_contacts):
                tmp = list(curr_labels[i])
                tmp[-1] = finger_combo[i]
                labels_tmp.append(tmp)
        return labels_tmp

    def get_real_contacts(self):
        collision_report = orpy.CollisionReport()
        real_contacts = []
        # iterate over all fingertip links and determine the contacts
        for eel in self._robot.get_fingertip_links():
            link = self._robot.GetLink(eel)
            self._orEnv.CheckCollision(self._obj, link, report=collision_report)
            # self._orEnv.CheckCollision(link, self._obj, report=collision_report)
            if len(collision_report.contacts) == 0:
                raise ValueError('[HFTSSampler::get_real_contacts] No contacts found')
            # TODO the normals reported by the collision check are wrong, so instead we use a nearest
            # TODO neighbor lookup. Should see what's wrong with OpenRAVE here...
            position = collision_report.contacts[0].pos
            normal = self._object_points[self._object_kd_tree.query(position), 3:][1]
            # normal = collision_report.contacts[0].norm
            real_contacts.append(np.concatenate((position, normal)))
        real_contacts = np.asarray(real_contacts)
        return real_contacts

    def get_root_node(self):
        possible_num_children, possible_num_leaves = self.get_branch_information(0)
        return HFTSNode(num_possible_children=possible_num_children,
                        num_possible_leaves=possible_num_leaves)

    def is_grasp_collision_free(self):
        links = self._robot.get_non_fingertip_links()
        for link in links:
            if self._orEnv.CheckCollision(self._robot.GetLink(link)):
                return False
        return True

    def load_hand(self, hand_file, hand_cache_file):
        if not self._hand_loaded:
            # TODO make this Robotiq hand independent (external hand loader)
            self._robot = RobotiqHand(hand_cache_file=hand_cache_file,
                                      env=self._orEnv, hand_file=hand_file)
            self._hand_manifold = self._robot.get_hand_manifold()
            self._hand_manifold.load()
            self._num_contacts = self._robot.get_contact_number()
            shift = transformations.identity_matrix()
            shift[0, -1] = 0.2
            self._robot.SetTransform(shift)
            rospy.loginfo('Hand loaded in OpenRAVE environment')
            self._hand_loaded = True

    def load_object(self, obj_id, model_id=None):
        if model_id is None:
            model_id = obj_id
        self._data_labeled, self._branching_factors, self._obj_com = \
            self._object_io_interface.get_hfts(model_id, self._b_force_new_hfts)
        if self._data_labeled is None:
            raise RuntimeError('Could not load HFTS model for model ' + model_id)
        self.create_object_kd_tree(self._data_labeled[:, :6])
        self._num_levels = len(self._branching_factors)
        # First, delete old object if there is any
        if self._obj_loaded:
            self._orEnv.Remove(self._obj)
        or_file_name = self._object_io_interface.get_openrave_file_name(model_id)
        self._obj_loaded = self._orEnv.Load(or_file_name)
        if not self._obj_loaded:
            raise RuntimeError('Could not load object model %s in OpenRAVE' % model_id)
        self._obj = self._orEnv.GetKinBody('objectModel')
        rospy.loginfo('Object loaded in OpenRAVE environment')
        if self._scene_interface is not None:
            self._scene_interface.set_target_object(obj_id)
        self.compute_contact_combinations()
        self._obj_loaded = True
        import IPython
        IPython.embed()

    def sample_grasp(self, node, depth_limit, post_opt=False, label_cache=None, open_hand_offset=0.1):
        if depth_limit < 0:
            raise ValueError('HFTSSampler::sample_grasp depth limit must be greater or equal to zero.')

        if node.get_depth() >= self._num_levels:
            raise ValueError('HFTSSampler::sample_grasp input node has an invalid depth')

        if node.get_depth() + depth_limit >= self._num_levels:
            depth_limit = self._num_levels - node.get_depth()  # cap

        # In case we using the integrated method, we might have a limitation on what nodes to descend to
        # let's compute this set.
        allowed_finger_combos = None
        if label_cache is not None and depth_limit == 1:
            # TODO This currently only works for hops == 2
            assert self._hops == 2
            allowed_finger_combos = self.compute_allowed_contact_combinations(node.get_depth(), label_cache)
            rospy.logdebug('[HFTSSampler::sample_grasp] We have %i allowed contacts' % len(allowed_finger_combos))
            if len(allowed_finger_combos) == 0:
                rospy.logwarn('[HFTSSampler::sample_grasp] We have no allowed contacts left! Aborting.')
                return node
        elif label_cache is not None and depth_limit != 1:
            raise ValueError('[HFTSSampler::sample_grasp] Label cache only works for depth_limit == 1')

        # Now, get a node to start stochastic optimization from
        seed_ik = None
        if node.get_depth() == 0: # at root
            contact_label = self.pick_new_start_node()
            best_o = -np.inf  # need to also consider non-root nodes
        else:
            # If we are not at a leaf node, go down in the hierarchy
            seed_ik = node.get_arm_configuration()
            contact_label = copy.deepcopy(node.get_labels())
            best_o, contact_label = self.extend_hfts_node(contact_label,
                                                          allowed_finger_combos=allowed_finger_combos)

        self.reset_robot()
        depth_limit -= 1
        rospy.logdebug('[HFTSSampler::sample_grasp] Sampling a grasp; %i number of iterations' % self._max_iters)
        # Do stochastic optimization until depth_limit is reached
        while depth_limit >= 0:
            # Randomly select siblings to optimize the objective function
            for iter_now in range(self._max_iters):
                labels_tmp = self.get_random_sibling_labels(curr_labels=contact_label,
                                                            allowed_finger_combos=allowed_finger_combos)
                s_tmp, r_tmp, o_tmp = self.evaluate_grasp(labels_tmp)
                if self.shc_evaluation(o_tmp, best_o):
                    contact_label = labels_tmp
                    best_o = o_tmp
                    # self._debug_visualize(labels_tmp, handle_index=0)
            # Descend to next level if we iterate at least once more
            if depth_limit > 0:
                best_o, contact_label = self.extend_hfts_node(contact_label)
            depth_limit -= 1

        # Evaluate grasp on robot hand
        # First, determine a hand configuration and the contact locations
        grasp_conf, object_contacts, hand_contacts = self.compose_grasp_info(contact_label)
        # Simulate the grasp and do local adjustments
        b_robotiq_ok, grasp_conf, grasp_pose = self.simulate_grasp(grasp_conf=grasp_conf,
                                                                   hand_contacts=hand_contacts,
                                                                   object_contacts=object_contacts,
                                                                   post_opt=post_opt,
                                                                   swap_contacts=label_cache is None)
        if b_robotiq_ok:
            sample_q = 0
            stability = best_o
        else:
            sample_q = 4
            stability = 0.0
        # except InvalidTriangleException:
        #     grasp_conf = None
        #     sample_q = 4
        #     stability = 0.0

        is_leaf = (len(contact_label[0]) == self._num_levels)
        is_goal_sample = (sample_q == 0) and is_leaf
        if not is_goal_sample and grasp_conf is not None:
            rospy.logdebug('[HFTSSampler::sample_grasp] Approximate has final quality: %i' % sample_q)
            b_approximate_feasible = self._robot.avoid_collision_at_fingers(n_step=20)
            if b_approximate_feasible:
                grasp_conf = self._robot.GetDOFValues()
                open_hand_offset = 0.0

        logging.debug('[HFTSSampler::sample_grasp] We sampled a grasp on level ' + str(len(contact_label[0])))
        if is_goal_sample:
            logging.debug('[HFTSSampler::sample_grasp] We sampled a goal grasp (might be in collision)!')
        if is_leaf:
            logging.debug('[HFTSSampler::sample_grasp] We sampled a leaf')

        if grasp_conf is not None and grasp_pose is not None:
            collision_free_arm_ik, arm_conf, pre_grasp_conf = \
                self.check_arm_grasp_validity(grasp_conf=grasp_conf,
                                              grasp_pose=grasp_pose,
                                              seed=seed_ik, open_hand_offset=open_hand_offset)
        else:
            collision_free_arm_ik = False
            arm_conf = None
            pre_grasp_conf = None

        depth = len(contact_label[0])
        possible_num_children, possible_num_leaves = self.get_branch_information(depth)
        return HFTSNode(labels=contact_label, hand_conf=np.asarray(grasp_conf),
                        pre_grasp_conf=pre_grasp_conf, arm_conf=arm_conf,
                        is_goal=is_goal_sample, is_leaf=is_leaf, is_valid=collision_free_arm_ik,
                        num_possible_children=possible_num_children, num_possible_leaves=possible_num_leaves,
                        hand_transform=self._robot.GetTransform())

    def set_max_iter(self, m):
        assert m > 0
        self._max_iters = m

    def set_parameters(self, max_iters=None, reachability_weight=None,
                       com_center_weight=None, hfts_generation_params=None,
                       b_force_new_hfts=None):
        # TODO some of these parameters are Robotiq hand specific. We probably wanna pass them as dictionary
        if max_iters is not None:
            self._max_iters = max_iters
            assert self._max_iters > 0
        if reachability_weight is not None:
            self._reachability_weight = reachability_weight
            assert self._reachability_weight >= 0.0
        # TODO this is Robotiq hand specific, and outdated
        self._hand_manifold.set_parameters(com_center_weight)
        if hfts_generation_params is not None:
            self._object_io_interface.set_hfts_generation_parameters(hfts_generation_params)
        if b_force_new_hfts is not None:
            self._b_force_new_hfts = b_force_new_hfts

    def shc_evaluation(self, o_tmp, best_o):
        if best_o < o_tmp:
            return True
        else:
            return False

    def _simulate_grasp(self, grasp_conf, hand_contacts, object_contacts, post_opt=False):
        # self.draw_contacts(object_contacts)
        self._robot.SetDOFValues(grasp_conf)
        try:
            T = self._robot.hand_obj_transform(hand_contacts[:3, :3], object_contacts[:, :3])
            self._robot.SetTransform(T)
        except InvalidTriangleException as ite:
            logging.warn('[HFTSSampler::simulate_grasp] Caught an InvalidTriangleException: ' + str(ite))
            return False, grasp_conf, None
        if post_opt:
            self._post_optimization(object_contacts)
        open_success, tips_in_contact = self._robot.comply_fingertips()
        if not open_success or not tips_in_contact:
            return False, self._robot.GetDOFValues(), self._robot.GetTransform()
        if self.check_grasp_validity():
            return True, self._robot.GetDOFValues(), self._robot.GetTransform()
        return False, self._robot.GetDOFValues(), self._robot.GetTransform()

    def simulate_grasp(self, grasp_conf, hand_contacts, object_contacts, post_opt=False, swap_contacts=True):
        # TODO this method as it is right now is only useful for the Robotiq hand.
        b_grasp_valid, grasp_conf, grasp_pose = self._simulate_grasp(grasp_conf, hand_contacts, object_contacts, post_opt)
        if not b_grasp_valid and swap_contacts:
            self.swap_contacts([0, 1], object_contacts)
            b_grasp_valid, grasp_conf, grasp_pose = self._simulate_grasp(grasp_conf, hand_contacts, object_contacts, post_opt)
        return b_grasp_valid, grasp_conf, grasp_pose

    @staticmethod
    def swap_contacts(rows, object_contacts):
        frm = rows[0]
        to = rows[1]
        object_contacts[[frm, to], :] = object_contacts[[to, frm], :]

    def reset_robot(self):
        shift = transformations.identity_matrix()
        shift[0, -1] = 0.2
        self._robot.SetTransform(shift)
        # Set hand to default (mean) configuration
        mean_values = map(lambda min_v, max_v: (min_v + max_v) / 2.0,
                          self._robot.GetDOFLimits()[0],
                          self._robot.GetDOFLimits()[1])
        self._robot.SetDOFValues(mean_values, range(len(mean_values)))

    def pick_new_start_node(self):
        num_nodes_top_level = self._branching_factors[0]
        contact_label = []
        for i in range(self._num_contacts):
            contact_label.append([np.random.choice(range(num_nodes_top_level + 1))])
        return contact_label

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

    def _post_optimization(self, grasp_contacts):
        logging.info('[HFTSSampler::_post_optimization] Performing post optimization.')
        transform = self._robot.GetTransform()
        angle, axis, point = transformations.rotation_from_matrix(transform)
        # further optimize hand configuration and pose
        # TODO this is Robotiq hand specific
        transform_params = axis.tolist() + [angle] + transform[:3, 3].tolist()
        robot_dofs = self._robot.GetDOFValues().tolist()

        def joint_limits_constraint(x, *args):
            positions, normals, robot = args
            lower_limits, upper_limits = robot.GetDOFLimits()
            return -dist_in_range(x[0], [lower_limits[0], upper_limits[0]]) - \
                   dist_in_range(x[1], [lower_limits[1], upper_limits[1]])

        def collision_free_constraint(x, *args):
            positions, normals, robot = args
            config = [x[0], x[1]]
            robot.SetDOFValues(config)
            env = robot.GetEnv()
            links = robot.get_non_fingertip_links()
            for link in links:
                if env.CheckCollision(robot.GetLink(link)):
                    return -1.0
            return 0.0

        x_min = scipy.optimize.fmin_cobyla(self._post_optimization_obj_fn, robot_dofs + transform_params,
                                           [joint_limits_constraint, collision_free_constraint],
                                           rhobeg=.5, rhoend=1e-3,
                                           args=(grasp_contacts[:, :3], grasp_contacts[:, 3:], self._robot),
                                           maxfun=int(1e8), iprint=0)
        self._robot.SetDOFValues(x_min[:2])
        axis = x_min[2:5]
        angle = x_min[5]
        position = x_min[6:]
        transform = transformations.rotation_matrix(angle, axis)
        transform[:3, 3] = position
        self._robot.SetTransform(transform)

    @staticmethod
    def _post_optimization_obj_fn(x, *params):
        # TODO this is Robotiq hand specific
        desired_contact_points, desired_contact_normals, robot = params
        dofs = x[:2]
        robot.SetDOFValues(dofs)
        axis = x[2:5]
        angle = x[5]
        position = x[6:]
        transform = transformations.rotation_matrix(angle, axis)
        transform[:3, 3] = position
        robot.SetTransform(transform)
        contacts = robot.get_tip_pn()
        temp_positions = contacts[:, :3]
        temp_normals = contacts[:, 3:]
        pos_err = position_distance(desired_contact_points, temp_positions)
        normal_err = normal_distance(desired_contact_normals, temp_normals)
        return pos_err + normal_err


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

