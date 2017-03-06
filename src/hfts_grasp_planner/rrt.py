#!/usr/bin/env python

""" This is a draft modification of the RRT algorithm for the sepcial case
    that sampling the goal region is computationally expensive """

import random
import numpy
import time
import math
import logging
import copy
from rtree import index


class SampleData:
    def __init__(self, config, data=None, data_copy_fn=copy.deepcopy, id_num=-1):
        self._config = config
        self._id = id_num
        self._data = data
        self._dataCopyFct = data_copy_fn

    def get_configuration(self):
        return self._config

    def get_data(self):
        return self._data

    def copy(self):
        copied_data = None
        if self._data is not None:
            copied_data = self._dataCopyFct(self._data)
        return SampleData(numpy.copy(self._config), copied_data, data_copy_fn=self._dataCopyFct, id_num=self._id)

    def is_valid(self):
        return self._config is not None

    def is_equal(self, other_sample_data):
        return (self._config == other_sample_data._config).all() and self._data == other_sample_data._data

    def get_id(self):
        return self._id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{SampleData:[Config=" + str(self._config) + "; Data=" + str(self._data) + "]}"


class TreeNode(object):
    def __init__(self, nid, pid, data):
        self._id = nid
        self._parent = pid
        self._data = data
        self._children = []

    def get_sample_data(self):
        return self._data

    def get_id(self):
        return self._id

    def get_parent_id(self):
        return self._parent

    def add_child_id(self, cid):
        self._children.append(cid)

    def get_children(self):
        return self._children

    def __str__(self):
        return "{TreeNode: [id=" + str(self._id) + ", Data=" + str(self._data) + "]}"


class Tree(object):
    TREE_ID = 0

    def __init__(self, root_data, b_forward_tree=True):
        self._nodes = [TreeNode(0, 0, root_data.copy())]
        self._labeled_nodes = []
        self._node_id = 1
        self._b_forward_tree = b_forward_tree
        self._tree_id = Tree.TREE_ID + 1
        Tree.TREE_ID += 1

    def add(self, parent, child_data):
        """
            Adds the given data as a child node of parent.
            @param parent: Must be of type TreeNode and denotes the parent node.
            @param child_data: SampleData that is supposed to be saved in the child node (it is copied).
        """
        child_node = TreeNode(self._node_id, parent.get_id(), child_data.copy())
        parent.add_child_id(child_node.get_id())
        self._nodes.append(child_node)
        # self._parents.append(parent.get_id())
        # assert(len(self._parents) == self._node_id + 1)
        self._node_id += 1
        return child_node

    def get_id(self):
        return self._tree_id

    def add_labeled_node(self, node):
        self._labeled_nodes.append(node)

    def get_labeled_nodes(self):
        return self._labeled_nodes

    def clear_labeled_nodes(self):
        self._labeled_nodes = []

    def remove_labeled_node(self, node):
        if node in self._labeled_nodes:
            self._labeled_nodes.remove(node)

    def nearest_neighbor(self, sample):
        pass

    def extract_path(self, goal_node):
        path = [goal_node.get_sample_data()]
        current_node = goal_node
        while current_node.get_id() != 0:
            current_node = self._nodes[current_node.get_parent_id()]
            path.append(current_node.get_sample_data())
        path.reverse()
        return path

    def get_root_node(self):
        return self._nodes[0]

    def size(self):
        return len(self._nodes)

    def merge(self, merge_node_a, other_tree, merge_node_b):
        """
            Merges this tree with the given tree. The connection is established through nodeA and nodeB,
            for which it is assumed that both nodeA and nodeB represent the same configuration.
            In other words, both the parent and all children of nodeB become children of nodeA.
            Labeled nodes of tree B will be added as labeled nodes of tree A.

            Runtime: O(size(otherTree) * num_labeled_nodes(otherTree))

            @param merge_node_a The node of this tree where to attach otherTree
            @param other_tree  The other tree (is not changed)
            @param merge_node_b The node of tree B that is merged with mergeNodeA from this tree.

            @return The root of treeB as a TreeNode of treeA after the merge.
        """
        node_stack = [(merge_node_a, merge_node_b, None)]
        b_root_node_in_a = None
        while len(node_stack) > 0:
            (current_node_a, current_node_b, ignore_id) = node_stack.pop()
            for child_id in current_node_b.get_children():
                if child_id == ignore_id:  # prevent adding duplicates
                    continue
                child_node_b = other_tree._nodes[child_id]
                child_node_a = self.add(current_node_a, child_node_b.get_sample_data())
                if child_node_b in other_tree._labeled_nodes:
                    self.add_labeled_node(child_node_a)
                node_stack.append((child_node_a, child_node_b, current_node_b.get_id()))

            # In case current_node_b is not the root of B, we also need to add the parent
            # as a child in this tree.
            parent_id = current_node_b.get_parent_id()
            if current_node_b.get_id() != parent_id:
                if parent_id != ignore_id:  # prevent adding duplicates
                    parent_node_b = other_tree._nodes[current_node_b.get_parent_id()]
                    child_node_a = self.add(current_node_a, parent_node_b.get_sample_data())
                    node_stack.append((child_node_a, parent_node_b, current_node_b.get_id()))
                    if parent_node_b in other_tree._labeled_nodes:
                        self.add_labeled_node(child_node_a)
            else:  # save the root to return it
                b_root_node_in_a = current_node_a

        return b_root_node_in_a


class SqrtTree(Tree):
    def __init__(self, root):
        super(SqrtTree, self).__init__(root)
        self.offset = 0

    def add(self, parent, child):
        child_node = super(SqrtTree, self).add(parent, child)
        self._update_stride()
        return child_node

    # def clear(self):
    #     super(SqrtTree, self).clear()
    #     self.offset = 0
    #     self.stride = 0

    def nearest_neighbor(self, q):
        """
            Computes an approximate nearest neighbor of q.
            To keep the computation time low, this method only considers sqrt(n)
            nodes, where n = #nodes.
            This implementation is essentially a copy from:
            http://ompl.kavrakilab.org/NearestNeighborsSqrtApprox_8h_source.html
            @return The tree node (Type TreeNode) for which the data point is closest to q.
        """
        d = float('inf')
        nn = None
        if self.stride > 0:
            for i in range(0, self.stride):
                pos = (i * self.stride + self.offset) % len(self._nodes)
                n = self._nodes[pos]
                dt = numpy.linalg.norm(q - n.get_sample_data().get_configuration())
                if dt < d:
                    d = dt
                    nn = n
            self.offset = random.randint(0, self.stride)
        return nn

    def _update_stride(self):
        self.stride = int(1 + math.floor(math.sqrt(len(self._nodes))))


class RTreeTree(Tree):
    def __init__(self, root, dimension, scaling_factors, b_forward_tree=True):
        super(RTreeTree, self).__init__(root, b_forward_tree=b_forward_tree)
        self._scaling_factors = scaling_factors
        self._create_index(dimension)
        self.dimension = dimension
        self._add_to_idx(self._nodes[0])

    def add(self, parent, child_data):
        child_node = super(RTreeTree, self).add(parent, child_data)
        self._add_to_idx(child_node)
        return child_node

    def nearest_neighbor(self, sample_data):
        if len(self._nodes) == 0:
            return None
        point_list = list(sample_data.get_configuration())
        point_list = map(lambda x, y: math.sqrt(x) * y, self._scaling_factors, point_list)
        point_list += point_list
        nns = list(self.idx.nearest(point_list))
        return self._nodes[nns[0]]

    def _add_to_idx(self, child_node):
        point_list = list(child_node.get_sample_data().get_configuration())
        point_list = map(lambda x, y: math.sqrt(x) * y, self._scaling_factors, point_list)
        point_list += point_list
        self.idx.insert(child_node.get_id(), point_list)

    def _create_index(self, dim):
        prop = index.Property()
        prop.dimension = dim
        self.idx = index.Index(properties=prop)


class Constraint(object):
    def project(self, old_config, config):
        return config


class ConstraintsManager(object):
    def __init__(self, callback_function=None):
        self._constraints_storage = {}
        self._active_constraints = []
        self._new_tree_callback = callback_function

    def project(self, old_config, config):
        if len(self._active_constraints) == 0:
            return config
        # For now we just iterate over all constraints and project successively
        for constraint in self._active_constraints:
            config = constraint.project(old_config, config)
        return config

    def set_active_tree(self, tree):
        if tree.get_id() in self._constraints_storage:
            self._active_constraints.extend(self._constraints_storage[tree.get_id()])

    def reset_constraints(self):
        self._active_constraints = []

    def clear(self):
        self._active_constraints = []
        self._constraints_storage = {}

    def register_new_tree(self, tree):
        if self._new_tree_callback is not None:
            self._constraints_storage[tree.get_id()] = self._new_tree_callback(tree)


class PGoalProvider(object):
    def compute_p_goal(self, num_trees):
        pass


class ConstPGoalProvider(PGoalProvider):
    def __init__(self, p_goal):
        self._pGoal = p_goal

    def compute_p_goal(self, num_trees):
        logging.debug('[ConstPGoalProvider::compute_p_goal] Returning constant pGoal')
        if num_trees == 0:
            return 1.0
        return self._pGoal


class DynamicPGoalProvider(PGoalProvider):
    def __init__(self, p_max=0.8, goal_w=1.2, p_goal_min=0.01):
        self._pMax = p_max
        self._goalW = goal_w
        self._pGoalMin = p_goal_min

    def compute_p_goal(self, num_trees):
        logging.debug('[DynamicPGoalProvider::compute_p_goal] Returning dynamic pGoal')
        return self._pMax * math.exp(-self._goalW * num_trees) + self._pGoalMin


class StatsLogger:
    def __init__(self):
        self.num_backward_trees = 0
        self.num_goals_sampled = 0
        self.num_valid_goals_sampled = 0
        self.num_approx_goals_sampled = 0
        self.num_attempted_tree_connects = 0
        self.num_successful_tree_connects = 0
        self.num_goal_nodes_sampled = 0
        self.num_c_free_samples = 0
        self.num_accumulated_logs = 1
        self.final_grasp_quality = 0.0
        self.runtime = 0.0
        self.success = 0
        self.treeSizes = {}

    def clear(self):
        self.num_backward_trees = 0
        self.num_goals_sampled = 0
        self.num_valid_goals_sampled = 0
        self.num_approx_goals_sampled = 0
        self.num_attempted_tree_connects = 0
        self.num_successful_tree_connects = 0
        self.num_goal_nodes_sampled = 0
        self.num_c_free_samples = 0
        self.num_accumulated_logs = 1
        self.final_grasp_quality = 0.0
        self.runtime = 0.0
        self.success = 0
        self.treeSizes = {}

    def to_dict(self):
        a_dict = {'numBackwardTrees': self.num_backward_trees, 'numGoalSampled': self.num_goals_sampled,
                  'numValidGoalSampled': self.num_valid_goals_sampled,
                  'numApproxGoalSampled': self.num_approx_goals_sampled,
                  'numGoalNodesSampled': self.num_goal_nodes_sampled,
                  'numSuccessfulTreeConnects': self.num_successful_tree_connects,
                  'numCFreeSamples': self.num_c_free_samples, 'finalGraspQuality': float(self.final_grasp_quality),
                  'runtime': self.runtime, 'success': self.success}
        return a_dict

    def print_logs(self):
        print 'Logs:'
        print '     num_backward_trees(avg): ', self.num_backward_trees
        print '     num_goals_sampled(avg): ', self.num_goals_sampled
        print '     num_valid_goals_sampled(avg): ', self.num_valid_goals_sampled
        print '     num_approx_goals_sampled(avg): ', self.num_approx_goals_sampled
        print '     num_goal_nodes_sampled(avg): ', self.num_goal_nodes_sampled
        print '     num_attempted_tree_connects(avg): ', self.num_attempted_tree_connects
        print '     num_successful_tree_connects(avg): ', self.num_successful_tree_connects
        print '     num_c_free_samples(avg): ', self.num_c_free_samples
        print '     final_grasp_quality(avg): ', self.final_grasp_quality
        print '     runtime(avg): ', self.runtime
        print '     success(avg): ', self.success
        if self.num_accumulated_logs == 1:
            print '     treeSizes: ', self.treeSizes

    def accumulate(self, other_logger):
        self.num_backward_trees += other_logger.numBackwardTrees
        self.num_goals_sampled += other_logger.numGoalSampled
        self.num_valid_goals_sampled += other_logger.numValidGoalSampled
        self.num_approx_goals_sampled += other_logger.numApproxGoalSampled
        self.num_attempted_tree_connects += other_logger.numAttemptedTreeConnects
        self.num_successful_tree_connects += other_logger.numSuccessfulTreeConnects
        self.num_goal_nodes_sampled += other_logger.numGoalNodesSampled
        self.num_c_free_samples += other_logger.numCFreeSamples
        self.num_accumulated_logs += other_logger.numAccumulatedLogs
        self.final_grasp_quality += other_logger.finalGraspQuality
        self.runtime += other_logger.runtime
        self.success += other_logger.success
        self.treeSizes.update(other_logger.treeSizes)

    def finalize_accumulation(self):
        self.num_backward_trees = float(self.num_backward_trees) / float(self.num_accumulated_logs)
        self.num_goals_sampled = float(self.num_goals_sampled) / float(self.num_accumulated_logs)
        self.num_valid_goals_sampled = float(self.num_valid_goals_sampled) / float(self.num_accumulated_logs)
        self.num_approx_goals_sampled = float(self.num_approx_goals_sampled) / float(self.num_accumulated_logs)
        self.num_attempted_tree_connects = float(self.num_attempted_tree_connects) / float(self.num_accumulated_logs)
        self.num_successful_tree_connects = float(self.num_successful_tree_connects) / float(self.num_accumulated_logs)
        self.num_goal_nodes_sampled = float(self.num_goal_nodes_sampled) / float(self.num_accumulated_logs)
        self.final_grasp_quality /= float(self.num_accumulated_logs)
        self.runtime /= float(self.num_accumulated_logs)
        self.success /= float(self.num_accumulated_logs)
        self.num_c_free_samples = float(self.num_c_free_samples) / float(self.num_accumulated_logs)


class RRT:
    def __init__(self, p_goal_provider, c_free_sampler, goal_sampler, logger, pgoal_tree=0.8,
                 constraints_manager=None):  # pForwardTree, pConnectTree
        """ Initializes the RRT planner
            @param pGoal - Instance of PGoalProvider that provides a probability of sampling a new goal
            @param c_free_sampler - A sampler of c_free.
            @param goal_sampler - A sampler of the goal region.
            @param logger - A logger (of type Logger) for printouts.
            @param constraints_manager - (optional) a constraint manager.
        """
        self.p_goal_provider = p_goal_provider
        self.p_goal_tree = pgoal_tree
        self.goal_sampler = goal_sampler
        self.c_free_sampler = c_free_sampler
        self.logger = logger
        self.stats_logger = StatsLogger()
        # self.debugConfigList = []
        if constraints_manager is None:
            constraints_manager = ConstraintsManager()
        self._constraints_manager = constraints_manager

    def extend(self, tree, random_sample, add_intermediates=True, add_tree_step=10):
        self._constraints_manager.set_active_tree(tree)
        nearest_node = tree.nearest_neighbor(random_sample)
        (bConnected, samples) = self.c_free_sampler.interpolate(nearest_node.get_sample_data(), random_sample,
                                                                projection_function=self._constraints_manager.project)
        parent_node = nearest_node
        self.logger.debug('[RRT::extend We have ' + str(len(samples) - 1) + " intermediate configurations")
        if add_intermediates:
            for i in range(add_tree_step, len(samples) - 1, add_tree_step):
                parent_node = tree.add(parent_node, samples[i].copy())
        if len(samples) > 1:
            last_node = tree.add(parent_node, samples[-1].copy())
        else:
            # self.debugConfigList.extend(samples)
            last_node = parent_node
        return last_node, bConnected

    def pick_nearest_tree(self, sample, backward_trees):
        nn = None
        dist = float('inf')
        tree = None
        for treeTemp in backward_trees:
            nn_temp = treeTemp.nearest_neighbor(sample)
            dist_temp = self.c_free_sampler.distance(sample.get_configuration(),
                                                     nn_temp.get_sample_data().get_configuration())
            if dist_temp < dist:
                dist = dist_temp
                nn = nn_temp
                tree = treeTemp
        return tree, nn

    def proximity_birrt(self, start_config, time_limit=60, debug_function=lambda x, y: None,
                        shortcut_time=5.0, timer_function=time.time):
        """ Bidirectional RRT algorithm with hierarchical goal region that
            uses free space proximity to bias sampling. """
        if not self.c_free_sampler.is_valid(start_config):
            self.logger.info('[RRT::proximityBiRRT] Start configuration is invalid. Aborting.')
            return None
        from sampler import FreeSpaceProximitySampler, FreeSpaceModel, ExtendedFreeSpaceModel
        assert type(self.goal_sampler) == FreeSpaceProximitySampler
        self.goal_sampler.clear()
        self.stats_logger.clear()
        self._constraints_manager.clear()
        # Create free space memories that our goal sampler needs
        connected_free_space = FreeSpaceModel(self.c_free_sampler)
        non_connected_free_space = ExtendedFreeSpaceModel(self.c_free_sampler)
        self.goal_sampler.set_connected_space(connected_free_space)
        self.goal_sampler.set_non_connected_space(non_connected_free_space)
        # Create forward tree
        forward_tree = RTreeTree(SampleData(start_config), self.c_free_sampler.get_space_dimension(),
                                 self.c_free_sampler.get_scaling_factors())
        self._constraints_manager.register_new_tree(forward_tree)
        connected_free_space.add_tree(forward_tree)
        # Various variable initializations
        backward_trees = []
        goal_tree_ids = []
        b_path_found = False
        path = None
        b_searching_forward = True
        # self.debugConfigList = []
        # Start
        start_time = timer_function()
        debug_function(forward_tree, backward_trees)

        # Main loop
        self.logger.debug('[RRT::proximityBiRRT] Starting planning loop')
        while timer_function() < start_time + time_limit and not b_path_found:
            debug_function(forward_tree, backward_trees)
            p = random.random()
            p_goal = self.p_goal_provider.compute_p_goal(len(backward_trees))
            self.logger.debug('[RRT::proximityBiRRT] Rolled a die: ' + str(p) + '. p_goal is ' +
                              str(p_goal))
            if p < p_goal:
                # Create a new backward tree
                self.logger.debug('[RRT::proximityBiRRT] Sampling a new goal configuration')
                goal_sample = self.goal_sampler.sample()
                self.stats_logger.num_goals_sampled += 1
                self.logger.debug('[RRT::proximityBiRRT] Sampled a new goal: ' + str(goal_sample))
                if goal_sample.is_valid():
                    backward_tree = RTreeTree(goal_sample, self.c_free_sampler.get_space_dimension(),
                                              self.c_free_sampler.get_scaling_factors(), b_forward_tree=False)
                    self._constraints_manager.register_new_tree(backward_tree)
                    if self.goal_sampler.is_goal(goal_sample):
                        self.stats_logger.num_valid_goals_sampled += 1
                        self.logger.debug('[RRT::proximityBiRRT] Goal sample is valid.' +
                                          ' Created new backward tree')
                        goal_tree_ids.append(backward_tree.get_id())
                    else:
                        self.stats_logger.num_approx_goals_sampled += 1
                        self.logger.debug('[RRT::proximityBiRRT] Goal sample is valid, but approximate.' +
                                          ' Created new approximate backward tree')
                    self.stats_logger.num_backward_trees += 1
                    backward_trees.append(backward_tree)
                    non_connected_free_space.add_tree(backward_tree)
            else:
                # Extend search trees
                self.logger.debug('[RRT::proximityBiRRT] Extending search trees')
                self._constraints_manager.reset_constraints()
                random_sample = self.c_free_sampler.sample()
                self.logger.debug('[RRT::proximityBiRRT] Drew random sample: ' + str(random_sample))
                self.stats_logger.num_c_free_samples += 1
                (forward_node, backward_node, backward_tree, b_connected) = (None, None, None, False)
                if b_searching_forward or len(backward_trees) == 0:
                    self.logger.debug('[RRT::proximityBiRRT] Extending forward tree to random sample')
                    (forward_node, b_connected) = self.extend(forward_tree, random_sample)
                    self.logger.debug('[RRT::proximityBiRRT] Forward tree connected to sample: ' + str(b_connected))
                    self.logger.debug('[RRT::proximityBiRRT] New forward tree node: ' + str(forward_node))
                    if len(backward_trees) > 0:
                        self.logger.debug('[RRT::proximityBiRRT] Attempting to connect forward tree ' +
                                          'to backward tree')
                        (backward_tree, nearest_node) = \
                            self.pick_nearest_tree(forward_node.get_sample_data(), backward_trees)
                        (backward_node, b_connected) = self.extend(backward_tree, forward_node.get_sample_data())
                    else:
                        b_connected = False
                else:
                    self.logger.debug('[RRT::proximityBiRRT] Extending backward tree to random sample')
                    # TODO try closest tree instead
                    backward_tree = self.pick_backward_tree(backward_trees,
                                                            goal_tree_ids)
                    # (backward_tree, nearest_node) = self._biRRT_helper_nearestTree(random_sample, backward_trees)
                    if backward_tree.get_id() in goal_tree_ids:
                        self.logger.debug('[RRT::proximityBiRRT] Attempting to connect goal tree!!!!')
                    (backward_node, b_connected) = self.extend(backward_tree, random_sample)
                    self.logger.debug('[RRT::proximityBiRRT] New backward tree node: ' + str(backward_node))
                    self.logger.debug('[RRT::proximityBiRRT] Backward tree connected to sample: ' +
                                      str(b_connected))
                    self.logger.debug('[RRT::proximityBiRRT] Attempting to connect forward tree ' +
                                      'to backward tree ' + str(backward_tree.get_id()))
                    (forward_node, b_connected) = self.extend(forward_tree, backward_node.get_sample_data())
                self.stats_logger.num_attempted_tree_connects += 1
                if b_connected:
                    self.logger.debug('[RRT::proximityBiRRT] Trees connected')
                    self.stats_logger.num_successful_tree_connects += 1
                    tree_name = 'merged_backward_tree' + str(self.stats_logger.num_successful_tree_connects)
                    self.stats_logger.treeSizes[tree_name] = backward_tree.size()
                    root_b = forward_tree.merge(forward_node, backward_tree, backward_node)
                    backward_trees.remove(backward_tree)
                    non_connected_free_space.remove_tree(backward_tree.get_id())
                    # Check whether we connected to a goal tree or not
                    if backward_tree.get_id() in goal_tree_ids:
                        goal_tree_ids.remove(backward_tree.get_id())
                        path = forward_tree.extract_path(root_b)
                        b_path_found = True
                        self.logger.debug('[RRT::proximityBiRRT] Found a path!')
                b_searching_forward = not b_searching_forward

        self.stats_logger.treeSizes['forward_tree'] = forward_tree.size()
        for bw_tree in backward_trees:
            self.stats_logger.treeSizes['unmerged_backward_tree' + str(bw_tree.get_id())] = bw_tree.size()
        debug_function(forward_tree, backward_trees)
        self.goal_sampler.debug_draw()
        self.stats_logger.num_goal_nodes_sampled = self.goal_sampler.get_num_goal_nodes_sampled()
        self.stats_logger.runtime = timer_function() - start_time
        if path is not None:
            self.stats_logger.final_grasp_quality = self.goal_sampler.get_quality(path[-1])
            self.stats_logger.success = 1
        return self.shortcut(path, shortcut_time)

    def pick_backward_tree(self, backward_trees, goal_tree_ids):
        p = random.random()
        goal_trees = [x for x in backward_trees if x.get_id() in goal_tree_ids]
        non_goal_trees = [x for x in backward_trees if x.get_id() not in goal_tree_ids]
        if p < self.p_goal_tree and len(goal_tree_ids) > 0:
            return random.choice(goal_trees)
        elif len(non_goal_trees) > 0:
            return random.choice(non_goal_trees)
        elif len(backward_trees) > 0:  # this may happen if we have only goal trees, but p >= self.pGoalTree
            return random.choice(backward_trees)
        else:
            raise ValueError('We do not have any backward trees to pick from')

    def shortcut(self, path, time_limit):
        if path is None:
            return None
        self.logger.debug('[RRT::shortcut] Shortcutting path of length %i with time limit %f' % (len(path),
                                                                                                 time_limit))
        start_time = time.clock()
        all_pairs = [(i, j) for i in range(len(path)) for j in range(i + 2, len(path))]
        random.shuffle(all_pairs)
        while time.clock() < start_time + time_limit and len(all_pairs) > 0:
            index_pair = all_pairs.pop()
            (bSuccess, samples) = self.c_free_sampler.interpolate(path[index_pair[0]], path[index_pair[1]])
            if bSuccess:
                path[index_pair[0] + 1:] = path[index_pair[1]:]
                all_pairs = [(i, j) for i in range(len(path)) for j in range(i + 2, len(path))]
                random.shuffle(all_pairs)
        self.logger.debug('[RRT::shortcut] Shorcutting finished. New path length %i' % len(path))
        return path
