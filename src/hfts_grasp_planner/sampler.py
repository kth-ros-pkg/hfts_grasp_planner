#!/usr/bin/env python
""" This module contains a general hierarchically organized goal region sampler. """

import ast
import logging
import math
import numpy
import random
from rtree import index
from rrt import SampleData

NUMERICAL_EPSILON = 0.00001


class SamplingResult:
    def __init__(self, configuration, hierarchy_info=None, data_extractor=None, cache_id=-1):
        self.configuration = configuration
        self.data_extractor = data_extractor
        self.hierarchy_info = hierarchy_info
        self.cache_id = cache_id

    def get_configuration(self):
        return self.configuration

    def to_sample_data(self):
        if self.data_extractor is not None:
            return SampleData(self.configuration, self.data_extractor.extractData(self.hierarchy_info),
                              self.data_extractor.getCopyFunction(), id_num=self.cache_id)
        return SampleData(self.configuration, id_num=self.cache_id)

    def is_valid(self):
        return self.hierarchy_info.is_valid()

    def is_goal(self):
        if self.hierarchy_info is not None:
            return self.hierarchy_info.is_goal()
        return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{SamplingResult:[Config=" + str(self.configuration) + "; Info=" + str(self.hierarchy_info) + "]}"


class CSpaceSampler:
    def __init__(self):
        pass

    def sample(self):
        pass

    def sample_gaussian_neighborhood(self, config, variance):
        pass

    def is_valid(self, qSample):
        pass

    def get_sampling_step(self):
        pass

    def get_space_dimension(self):
        return 0

    def get_upper_bounds(self):
        pass

    def get_lower_bounds(self):
        pass

    def get_scaling_factors(self):
        return self.get_space_dimension() * [1]

    def distance(self, config_a, config_b):
        total_sum = 0.0
        scaling_factors = self.get_scaling_factors()
        for i in range(len(config_a)):
            total_sum += scaling_factors[i] * math.pow(config_a[i] - config_b[i], 2)
        return math.sqrt(total_sum)

    def configs_are_equal(self, config_a, config_b):
        dist = self.distance(config_a, config_b)
        return dist < NUMERICAL_EPSILON

    def interpolate(self, start_sample, end_sample, projection_function=lambda x, y: y):
        """
        Samples cspace linearly from the startSample to endSample until either
        a collision is detected or endSample is reached. All intermediate sampled configurations
        are returned in a list as SampleData.
        If a projectionFunction is specified, each sampled configuration is projected using this
        projection function. This allows to interpolate within a constraint manifold, i.e. some subspace of
        the configuration space. Additionally to the criterias above, the method also terminates when
        no more progress is made towards endSample.
        @param start_sample        The SampleData to start from.
        @param end_sample          The SampleData to sample to.
        @param projection_function (Optional) A projection function on a contraint manifold.
        @return A tuple (bSuccess, samples), where bSuccess is True if a connection to endSample was found;
                samples is a list of all intermediate sampled configurations [startSample, ..., lastSampled].
        """
        waypoints = [start_sample]
        config_sample = start_sample.get_configuration()
        pre_config_sample = start_sample.get_configuration()
        dist_to_target = self.distance(end_sample.get_configuration(), config_sample)
        while True:
            pre_dist_to_target = dist_to_target
            dist_to_target = self.distance(end_sample.get_configuration(), config_sample)
            if self.configs_are_equal(config_sample, end_sample.get_configuration()):
                # We reached our target. Since we want to keep data stored in the target, simply replace
                # the last waypoint with the instance endSample
                waypoints.pop()
                waypoints.append(end_sample)
                return True, waypoints
            elif dist_to_target > pre_dist_to_target:
                # The last sample we added, took us further away from the target, then the previous one.
                # Hence remove the last sample and return.
                waypoints.pop()
                return False, waypoints
            # We are still making progress, so sample a new sample
            # To prevent numerical issues, we move at least NUMERICAL_EPSILON
            step = min(self.get_sampling_step(), max(dist_to_target, NUMERICAL_EPSILON))
            config_sample = config_sample + step * (end_sample.get_configuration() - config_sample) / dist_to_target
            # Project the sample to the constraint manifold
            config_sample = projection_function(pre_config_sample, config_sample)
            if config_sample is not None and self.is_valid(config_sample):
                # We have a new valid sample, so add it to the waypoints list
                waypoints.append(SampleData(numpy.copy(config_sample)))
                pre_config_sample = numpy.copy(config_sample)
            else:
                # We ran into an obstacle - we won t get further, so just return what we have so far
                return False, waypoints


class SimpleHierarchyNode:
    class DummyHierarchyInfo:
        def __init__(self, unique_label):
            self.unique_label = unique_label

        def get_unique_label(self):
            return self.unique_label

        def is_goal(self):
            return False

        def is_valid(self):
            return False

    def __init__(self, config, hierarchy_info):
        self.config = config
        self.hierarchy_info = hierarchy_info
        self.children = []

    def get_children(self):
        return self.children

    def get_active_children(self):
        return self.children

    def get_num_children(self):
        return len(self.children)

    def get_max_num_children(self):
        return 1  # self.hierarchy_info.getPossibleNumChildren()

    def get_num_leaves_in_branch(self):
        return 0

    def get_max_num_leaves_in_branch(self):
        return 1  # self.hierarchy_info.getPossibleNumLeaves()

    def get_unique_label(self):
        return self.hierarchy_info.get_unique_label()

    def get_T(self):
        if self.hierarchy_info.is_goal() and self.hierarchy_info.is_valid():
            return 1.5
        if self.hierarchy_info.is_valid():
            return 1.0
        return 0.0

    def is_goal(self):
        return self.hierarchy_info.is_goal()

    def get_active_configuration(self):
        return self.config

    def add_child(self, child):
        self.children.append(child)


class NaiveGoalSampler:
    def __init__(self, goal_region, num_iterations=40, debug_drawer=None):
        self.goal_region = goal_region
        self.depth_limit = goal_region.get_max_depth()
        self.goal_region.set_max_iter(num_iterations)
        self._debug_drawer = debug_drawer
        self.clear()

    def clear(self):
        self.cache = []
        self._root_node = SimpleHierarchyNode(None, self.goal_region.get_root())
        self._label_node_map = {}
        self._label_node_map['root'] = self._root_node
        if self._debug_drawer is not None:
            self._debug_drawer.clear()

    def get_num_goal_nodes_sampled(self):
        return len(self._label_node_map)

    def _compute_ancestor_labels(self, unique_label):
        label_as_list = ast.literal_eval(unique_label)
        depth = len(label_as_list) / 3
        n_depth = depth - 1
        ancestor_labels = []
        while n_depth >= 1:
            ancestor_label = []
            for f in range(3):
                ancestor_label.extend(label_as_list[f * depth:f * depth + n_depth])
            ancestor_labels.append(str(ancestor_label))
            label_as_list = ancestor_label
            depth -= 1
            n_depth = depth - 1
        ancestor_labels.reverse()
        return ancestor_labels

    def _add_new_sample(self, sample):
        unique_label = sample.hierarchy_info.get_unique_label()
        ancestor_labels = self._compute_ancestor_labels(unique_label)
        parent = self._root_node
        for ancestorLabel in ancestor_labels:
            if ancestorLabel in self._label_node_map:
                parent = self._label_node_map[ancestorLabel]
            else:
                ancestor_node = SimpleHierarchyNode(config=None,
                                                    hierarchy_info=SimpleHierarchyNode.DummyHierarchyInfo(
                                                        ancestorLabel))
                parent.add_child(ancestor_node)
                self._label_node_map[ancestorLabel] = ancestor_node
                parent = ancestor_node
        if unique_label in self._label_node_map:
            return
        new_node = SimpleHierarchyNode(config=sample.get_configuration(), hierarchy_info=sample.hierarchy_info)
        self._label_node_map[unique_label] = new_node
        parent.add_child(new_node)

    def sample(self, b_dummy=False):
        logging.debug('[NaiveGoalSampler::sample] Sampling a goal in the naive way')
        my_sample = self.goal_region.sample(self.depth_limit)
        self._add_new_sample(my_sample)
        if self._debug_drawer is not None:
            self._debug_drawer.draw_hierarchy(self._root_node)
        if not my_sample.is_valid() or not my_sample.is_goal():
            logging.debug('[NaiveGoalSampler::sample] Failed. Did not get a valid goal!')
            return SampleData(None)
        else:
            my_sample.cacheId = len(self.cache)
            self.cache.append(my_sample)

        logging.debug('[NaiveGoalSampler::sample] Success. Found a valid goal!')
        return my_sample.to_sample_data()

    def get_quality(self, sample_data):
        idx = sample_data.get_id()
        return self.cache[idx].hierarchy_info.get_quality()

    def is_goal(self, sample):
        sampled_before = 0 < sample.get_id() < len(self.cache)
        if sampled_before:
            return self.goal_region.is_goal(self.cache[sample.get_id()])
        return False


class FreeSpaceModel(object):
    def __init__(self, c_space_sampler):
        self._trees = []
        self._c_space_sampler = c_space_sampler

    def add_tree(self, tree):
        self._trees.append(tree)

    def remove_tree(self, tree_id):
        new_tree_list = []
        for tree in self._trees:
            if tree.get_id() != tree_id:
                new_tree_list.append(tree)
        self._trees = new_tree_list

    def get_nearest_configuration(self, config):
        (dist, nearest_config) = (float('inf'), None)
        for tree in self._trees:
            tree_node = tree.nearest_neighbor(SampleData(config))
            tmp_config = tree_node.get_sample_data().get_configuration()
            tmp_dist = self._c_space_sampler.distance(config, tmp_config)
            if tmp_dist < dist:
                dist = tmp_dist
                nearest_config = tmp_config
        return dist, nearest_config


class ExtendedFreeSpaceModel(FreeSpaceModel):
    def __init__(self, c_space_sampler):
        super(ExtendedFreeSpaceModel, self).__init__(c_space_sampler)
        self._scaling_factors = c_space_sampler.get_scaling_factors()
        prop = index.Property()
        prop.dimension = c_space_sampler.get_space_dimension()
        self._approximate_index = index.Index(properties=prop)
        self._approximate_configs = []
        self._temporal_mini_cache = []

    def get_nearest_configuration(self, config):
        (tree_dist, nearest_tree_config) = super(ExtendedFreeSpaceModel, self).get_nearest_configuration(config)
        (temp_dist, temp_config) = self.get_closest_temporary(config)
        if temp_dist < tree_dist:
            tree_dist = temp_dist
            nearest_tree_config = temp_config

        if len(self._approximate_configs) > 0:
            point_list = self._make_coordinates(config)
            nns = list(self._approximate_index.nearest(point_list))
            nn_id = nns[0]
            nearest_approximate = self._approximate_configs[nn_id]
            assert nearest_approximate is not None
            dist = self._c_space_sampler.distance(nearest_approximate, config)
            if nearest_tree_config is not None:
                if tree_dist <= dist:
                    return tree_dist, nearest_tree_config
            return dist, nearest_approximate
        elif nearest_tree_config is not None:
            return tree_dist, nearest_tree_config
        else:
            return float('inf'), None

    def add_temporary(self, configs):
        self._temporal_mini_cache.extend(configs)

    def clear_temporary_cache(self):
        self._temporal_mini_cache = []

    def get_closest_temporary(self, config):
        min_dist, closest = float('inf'), None
        for tconfig in self._temporal_mini_cache:
            tdist = self._c_space_sampler.distance(config, tconfig)
            if tdist < min_dist:
                min_dist = tdist
                closest = tconfig
        return min_dist, closest

    def add_approximate(self, config):
        cid = len(self._approximate_configs)
        self._approximate_configs.append(config)
        point_list = self._make_coordinates(config)
        self._approximate_index.insert(cid, point_list)

    def draw_random_approximate(self):
        idx = len(self._approximate_configs) - 1
        if idx == -1:
            return None
        config = self._approximate_configs.pop()
        assert config is not None
        self._approximate_index.delete(idx, self._make_coordinates(config))
        return config

    def _make_coordinates(self, config):
        point_list = list(config)
        point_list = map(lambda x, y: math.sqrt(x) * y, self._scaling_factors, point_list)
        point_list += point_list
        return point_list


class FreeSpaceProximityHierarchyNode(object):
    def __init__(self, goal_node, config=None, initial_temp=0.0, active_children_capacity=20):
        self._goal_nodes = []
        self._goal_nodes.append(goal_node)
        self._active_goal_node_idx = 0
        self._children = []
        self._children_contact_labels = []
        self._active_children = []
        self._inactive_children = []
        self._t = initial_temp
        self._T = 0.0
        self._T_c = 0.0
        self._T_p = 0.0
        self._num_leaves_in_branch = 1 if goal_node.is_leaf() else 0
        self._active_children_capacity = active_children_capacity
        self._configs = []
        self._configs.append(None)
        self._configs_registered = []
        self._configs_registered.append(True)
        self._parent = None
        # INVARIANT: _configs[0] is always None
        #            _goal_nodes[0] is hierarchy node that has all information
        #            _configs_registered[i] is False iff _configs[i] is valid and new
        #            _goal_nodes[i] and _configs[i] belong together for i > 0
        if config is not None:
            self._goal_nodes.append(goal_node)
            self._configs.append(config)
            self._active_goal_node_idx = 1
            self._configs_registered.append(not goal_node.is_valid())

    def get_T(self):
        return self._T

    def get_t(self):
        return self._t

    def get_T_c(self):
        return self._T_c

    def get_T_p(self):
        return self._T_p

    def set_T(self, value):
        self._T = value
        assert self._T > 0.0

    def set_t(self, value):
        self._t = value
        assert self._t > 0.0 or self.is_leaf() and not self.has_configuration()

    def set_T_p(self, value):
        self._T_p = value

    def set_T_c(self, value):
        self._T_c = value
        assert self._T_c > 0.0

    def update_active_children(self, up_temperature_fn):
        # For completeness, reactivate a random inactive child:
        if len(self._inactive_children) > 0:
            reactivated_child = self._inactive_children.pop()
            up_temperature_fn(reactivated_child)
            self._active_children.append(reactivated_child)

        while len(self._active_children) > self._active_children_capacity:
            p = random.random()
            sum_temp = 0.0
            for child in self._active_children:
                sum_temp += 1.0 / child.get_T_c()

            assert sum_temp > 0.0
            acc = 0.0
            i = 0
            while acc < p:
                acc += 1.0 / self._active_children[i].get_T_c() * 1.0 / sum_temp
                i += 1
            deleted_child = self._active_children[max(i - 1, 0)]
            self._active_children.remove(deleted_child)
            self._inactive_children.append(deleted_child)
            logging.debug('[FreeSpaceProximityHierarchyNode::updateActiveChildren] Removing child with ' + \
                          'temperature ' + str(deleted_child.get_T()) + '. It had index ' + str(i))
        assert len(self._children) == len(self._inactive_children) + len(self._active_children)

    def add_child(self, child):
        self._children.append(child)
        self._active_children.append(child)
        self._children_contact_labels.append(child.get_contact_labels())
        child._parent = self
        if child.is_leaf():
            self._num_leaves_in_branch += 1
            parent = self._parent
            while parent is not None:
                parent._num_leaves_in_branch += 1
                parent = parent._parent

    def get_num_leaves_in_branch(self):
        return self._num_leaves_in_branch

    def get_max_num_leaves_in_branch(self):
        return self._goal_nodes[0].get_num_possible_leaves()

    def get_max_num_children(self):
        return self._goal_nodes[0].get_num_possible_children()

    def get_coverage(self):
        if self.is_leaf():
            return 1.0
        return self.get_num_children() / float(self.get_max_num_children())

    def get_parent(self):
        return self._parent

    def get_quality(self):
        return self._goal_nodes[0].get_quality()

    def has_children(self):
        return self.get_num_children() > 0

    def get_num_children(self):
        return len(self._children)

    def get_children(self):
        return self._children

    def get_contact_labels(self):
        return self._goal_nodes[0].get_labels()

    def get_children_contact_labels(self):
        return self._children_contact_labels

    def get_active_children(self):
        return self._active_children

    def get_unique_label(self):
        return self._goal_nodes[0].get_unique_label()

    def get_configurations(self):
        """ Returns all configurations stored for this hierarchy node."""
        return self._configs[1:]

    def get_valid_configurations(self):
        """ Returns only valid configurations """
        valid_configs = []
        for idx in range(1, len(self._goal_nodes)):
            if self._goal_nodes[idx].is_valid():
                valid_configs.append(self._configs[idx])
        return valid_configs

    def get_depth(self):
        return self._goal_nodes[0].get_depth()

    def is_root(self):
        return self._goal_nodes[0].get_depth() == 0

    def set_active_configuration(self, idx):
        assert idx in range(len(self._goal_nodes) - 1)
        self._active_goal_node_idx = idx + 1

    def get_active_configuration(self):
        if self._active_goal_node_idx in range(len(self._configs)):
            return self._configs[self._active_goal_node_idx]
        else:
            return None

    def has_configuration(self):
        return len(self._configs) > 1

    def get_new_valid_configs(self):
        unregistered_goals = []
        unregistered_approx = []
        for i in range(1, len(self._configs)):
            if not self._configs_registered[i]:
                if self._goal_nodes[i].is_goal():
                    unregistered_goals.append((self._configs[i], self._goal_nodes[i].get_hand_config()))
                else:
                    unregistered_approx.append(self._configs[i])
                self._configs_registered[i] = True
        return unregistered_goals, unregistered_approx

    def is_goal(self):
        return self._goal_nodes[self._active_goal_node_idx].is_goal() and self.is_valid()

    def is_valid(self):
        b_is_valid = self._goal_nodes[self._active_goal_node_idx].is_valid()
        # TODO had to remove the following assertion. If the grasp optimization is non deterministic
        # TODO it can happen that a result is once invalid and once valid. However, the label_cache
        # TODO should prevent this from happening
        # if not b_is_valid:
            # assert not reduce(lambda x, y: x or y, [x.is_valid() for x in self._goal_nodes], False)
        return self._goal_nodes[self._active_goal_node_idx].is_valid()

    def is_extendible(self):
        return self._goal_nodes[0].is_extendible()

    def is_leaf(self):
        return not self.is_extendible()

    def is_all_covered(self):
        if self.is_leaf():
            return True
        return self.get_num_children() == self.get_max_num_children()

    def get_goal_sampler_hierarchy_node(self):
        return self._goal_nodes[self._active_goal_node_idx]

    def to_sample_data(self, id_num=-1):
        return SampleData(self._configs[self._active_goal_node_idx],
                          data=self._goal_nodes[self._active_goal_node_idx].get_hand_config(),
                          id_num=id_num)

    def add_goal_sample(self, sample):
        sample_config = sample.get_configuration()
        if sample_config is None:
            return
        for config in self._configs[1:]:
            b_config_known = numpy.linalg.norm(config - sample_config) < NUMERICAL_EPSILON
            if b_config_known:
                return
        self._configs.append(sample.get_configuration())
        self._goal_nodes.append(sample.hierarchy_info)
        self._configs_registered.append(not sample.is_valid())


class FreeSpaceProximitySampler(object):
    def __init__(self, goal_sampler, c_free_sampler, k=4, num_iterations=10,
                 min_num_iterations=8,
                 b_return_approximates=True,
                 connected_weight=10.0, free_space_weight=5.0, debug_drawer=None):
        self._goal_hierarchy = goal_sampler
        self._k = k
        # if numIterations is None:
        #     numIterations = goalSampler.getMaxDepth() * [10]
        # elif type(numIterations) == int:
        #     numIterations = goalSampler.getMaxDepth() * [numIterations]
        # elif type(numIterations) == list:
        #     numIterations = numIterations
        # else:
        #     raise ValueError('numIterations has invalid type %s. Supported are int, list and None' %
        #                      str(type(numIterations)))
        # TODO decide how we wanna do this properly. Should the user be able to define level specific num iterations?
        self._num_iterations = max(1, goal_sampler.get_max_depth()) * [num_iterations]
        self._min_num_iterations = min_num_iterations
        self._connected_weight = connected_weight
        self._free_space_weight = free_space_weight
        self._connected_space = None
        self._non_connected_space = None
        self._debug_drawer = debug_drawer
        self._c_free_sampler = c_free_sampler
        self._label_cache = {}
        self._goal_labels = []
        self._root_node = FreeSpaceProximityHierarchyNode(goal_node=self._goal_hierarchy.get_root(),
                                                          initial_temp=self._free_space_weight)
        max_dist = numpy.linalg.norm(c_free_sampler.get_upper_bounds() - c_free_sampler.get_lower_bounds())
        self._min_connection_chance = self._distance_kernel(max_dist)
        self._min_free_space_chance = self._distance_kernel(max_dist)
        self._b_return_approximates = b_return_approximates

    def clear(self):
        logging.debug('[FreeSpaceProximitySampler::clear] Clearing caches etc')
        self._connected_space = None
        self._non_connected_space = None
        self._label_cache = {}
        self._goal_labels = []
        self._root_node = FreeSpaceProximityHierarchyNode(goal_node=self._goal_hierarchy.get_root(),
                                                          initial_temp=self._free_space_weight)
        self._num_iterations = self._goal_hierarchy.get_max_depth() * [self._num_iterations[0]]
        if self._debug_drawer is not None:
            self._debug_drawer.clear()

    def get_num_goal_nodes_sampled(self):
        return len(self._label_cache)

    def get_quality(self, sample_data):
        idx = sample_data.get_id()
        node = self._label_cache[self._goal_labels[idx]]
        return node.get_quality()

    def set_connected_space(self, connected_space):
        self._connected_space = connected_space

    def set_non_connected_space(self, non_connected_space):
        self._non_connected_space = non_connected_space

    def set_parameters(self, min_iterations=None, max_iterations=None,
                       free_space_weight=None, connected_space_weight=None,
                       use_approximates=None, k=None):
        if min_iterations is not None:
            self._min_num_iterations = min_iterations
        if max_iterations is not None:
            max_iterations = max(self._min_num_iterations, max_iterations)
            self._num_iterations = max(1, self._goal_hierarchy.get_max_depth()) * [max_iterations]
        if free_space_weight is not None:
            self._free_space_weight = free_space_weight
        if connected_space_weight is not None:
            self._connected_weight = connected_space_weight
        if use_approximates is not None:
            self._b_return_approximates = use_approximates
        if k is not None:
            self._k = k

    def _get_hierarchy_node(self, goal_sample):
        label = goal_sample.hierarchy_info.get_unique_label()
        b_new = False
        hierarchy_node = None
        if label in self._label_cache:
            hierarchy_node = self._label_cache[label]
            hierarchy_node.add_goal_sample(goal_sample)
            logging.warn('[FreeSpaceProximitySampler::_getHierarchyNode] Sampled a cached node!')
        else:
            hierarchy_node = FreeSpaceProximityHierarchyNode(goal_node=goal_sample.hierarchy_info,
                                                             config=goal_sample.get_configuration())
            self._label_cache[label] = hierarchy_node
            b_new = True
        return hierarchy_node, b_new

    def _filter_redundant_children(self, children):
        labeled_children = []
        filtered_children = []
        for child in children:
            labeled_children.append((child.get_unique_label(), child))
        labeled_children.sort(key=lambda x: x[0])
        prev_label = ''
        for labeledChild in labeled_children:
            if labeledChild[0] == prev_label:
                continue
            filtered_children.append(labeledChild[1])
            prev_label = labeledChild[0]
        return filtered_children

    def _compute_connection_chance(self, config):
        (dist, nearest_config) = self._connected_space.get_nearest_configuration(config)
        if nearest_config is None:
            return self._min_connection_chance
        return self._distance_kernel(dist)

    def _compute_free_space_chance(self, config):
        (dist, nearest_config) = self._non_connected_space.get_nearest_configuration(config)
        if nearest_config is None:
            return self._min_free_space_chance
        return self._distance_kernel(dist)

    def _distance_kernel(self, dist):
        return math.exp(-dist)

    def _update_temperatures(self, node):
        logging.debug('[FreeSpaceProximitySampler::_updateTemperatures] Updating temperatures')
        self._T(node)

    def _t(self, node):
        if node.is_root():
            node.set_t(self._free_space_weight)
            return self._free_space_weight
        if not node.has_configuration() and node.is_extendible():
            parent = node.get_parent()
            assert parent is not None
            tN = parent.get_coverage() * parent.get_t()
            node.set_t(tN)
            return tN
        elif not node.has_configuration() and node.is_leaf():
            # TODO: we should actually set this to 0.0 and prune covered useless branches
            minimal_temp = self._min_connection_chance + self._min_free_space_chance
            node.set_t(minimal_temp)
            return minimal_temp
        max_temp = 0.0
        config_id = 0
        for config in node.get_configurations():
            connected_temp = self._connected_weight * self._compute_connection_chance(config)
            free_space_temp = self._free_space_weight * self._compute_free_space_chance(config)
            temp = connected_temp + free_space_temp
            if max_temp < temp:
                node.set_active_configuration(config_id)
                max_temp = temp
            config_id += 1
        node.set_t(max_temp)
        # if not ((node.is_valid() and max_temp >= self._free_space_weight) or not node.is_valid()):
        #     print "WTF Assertion fail here"
        #     import IPython
        #     IPython.embed()
        #TODO this assertion failed once. This indicates a serious bug, but did not manage to reproduce it!
        assert not node.is_valid() or max_temp >= self._free_space_weight
        return node.get_t()

    def _T(self, node):
        temps_children = 0.0
        t_node = self._t(node)
        avg_child_temp = t_node
        if len(node.get_active_children()) > 0:
            for child in node.get_active_children():
                temps_children += self._T(child)
            avg_child_temp = temps_children / float(len(node.get_active_children()))
        node.set_T((t_node + avg_child_temp) / 2.0)
        self._T_c(node)
        self._T_p(node)
        return node.get_T()

    def _T_c(self, node):
        mod_branch_coverage = node.get_num_leaves_in_branch() / (node.get_max_num_leaves_in_branch() + 1)
        T_c = node.get_T() * (1.0 - mod_branch_coverage)
        node.set_T_c(T_c)
        return T_c

    def _T_p(self, node):
        T_p = node.get_T() * (1.0 - node.get_coverage())
        node.set_T_p(T_p)
        return T_p

    def _pick_random_node(self, p, nodes):
        modified_temps = [self._T_c(x) for x in nodes]
        acc_temp = sum(modified_temps)
        assert acc_temp > 0.0
        i = 0
        acc = 0.0
        while p > acc:
            acc += modified_temps[i] / acc_temp
            i += 1

        idx = max(i - 1, 0)
        other_nodes = nodes[:idx]
        if idx + 1 < len(nodes):
            other_nodes.extend(nodes[idx + 1:])
        return nodes[idx], other_nodes

    def _update_approximate(self, children):
        for child in children:
            goal_configs, approx_configs = child.get_new_valid_configs()
            assert len(goal_configs) == 0
            for config in approx_configs:
                self._non_connected_space.add_approximate(config)

    def _pick_random_approximate(self):
        random_approximate = self._non_connected_space.draw_random_approximate()
        logging.debug('[FreeSpaceProximitySampler::_pickRandomApproximate] ' + str(random_approximate))
        return random_approximate

    def _add_temporary(self, children):
        for node in children:
            if node.is_valid():
                self._non_connected_space.add_temporary(node.get_valid_configurations())

    def _clear_temporary(self):
        self._non_connected_space.clear_temporary_cache()

    def _pick_random_child(self, node):
        if not node.has_children():
            return None
        node.update_active_children(self._update_temperatures)
        p = random.random()
        (child, otherChildren) = self._pick_random_node(p, node.get_active_children())
        return child

    def _should_descend(self, parent, child):
        if child is None:
            return False
        if not child.is_extendible():
            return False
        if parent.is_all_covered():
            return True
        p = random.random()
        tP = self._T_p(parent)
        sum_temp = tP + self._T_c(child)
        if p <= tP / sum_temp:
            return False
        return True

    def _sample_child(self, node):
        goal_node = node.get_goal_sampler_hierarchy_node()
        depth = node.get_depth()
        num_iterations = int(self._min_num_iterations + \
                             node.get_T() / (self._connected_weight + self._free_space_weight) * \
                             (self._num_iterations[depth] - self._min_num_iterations))
        # num_iterations = max(self._minNumIterations, int(num_iterations))
        assert num_iterations >= self._min_num_iterations
        assert num_iterations <= self._num_iterations[depth]
        self._goal_hierarchy.set_max_iter(num_iterations)
        do_post_opt = depth == self._goal_hierarchy.get_max_depth() - 1
        children_contact_labels = node.get_children_contact_labels()
        goal_sample = self._goal_hierarchy.sample_warm_start(hierarchy_node=goal_node, depth_limit=1,
                                                             label_cache=children_contact_labels,
                                                             post_opt=do_post_opt)
        if goal_sample.hierarchy_info.is_goal() and goal_sample.hierarchy_info.is_valid():
            logging.debug('[FreeSpaceProximitySampler::_sampleChild] We sampled a valid goal here!!!')
        elif goal_sample.hierarchy_info.is_valid():
            logging.debug('[FreeSpaceProximitySampler::_sampleChild] Valid sample here!')
        (hierarchy_node, b_new) = self._get_hierarchy_node(goal_sample)
        if b_new:
            node.add_child(hierarchy_node)
        else:
            assert hierarchy_node.get_unique_label() == node.get_unique_label()
        return hierarchy_node

    def is_goal(self, sample):
        if sample.get_id() >= 0:
            return True
        return False

    def sample(self):
        current_node = self._root_node
        logging.debug('[FreeSpaceProximitySampler::sample] Starting to sample a new goal candidate' + \
                      ' - the lazy way')
        num_samplings = self._k
        b_temperatures_invalid = True
        while num_samplings > 0:
            if self._debug_drawer is not None:
                self._debug_drawer.draw_hierarchy(self._root_node)
            logging.debug('[FreeSpaceProximitySampler::sample] Picking random cached child')
            if b_temperatures_invalid:
                self._update_temperatures(current_node)
            child = self._pick_random_child(current_node)
            if self._should_descend(current_node, child):
                current_node = child
                b_temperatures_invalid = False
            elif current_node.is_all_covered():
                # There are no children left to sample, sample the null space of the child instead
                # TODO: It depends on our IK solver on what we have to do here. If the IK solver is complete,
                # we do not resample non-goal nodes. If it is not complete, we would need to give them
                # another chance here. Hence, we would also need to set the temperatures of such nodes
                # to sth non-zero
                if child.is_goal():
                    logging.warn('[FreeSpaceProximitySampler::sample] Pretending to sample null space')
                    # TODO actually sample null space here and return new configuration or approx
                    b_temperatures_invalid = True
                num_samplings -= 1
            else:
                new_child = self._sample_child(current_node)
                b_temperatures_invalid = True
                num_samplings -= 1
                goal_configs, approx_configs = new_child.get_new_valid_configs()
                assert len(goal_configs) + len(approx_configs) <= 1
                # if new_child.is_valid() and new_child.is_goal():
                if len(goal_configs) > 0:
                    self._goal_labels.append(new_child.get_unique_label())
                    return SampleData(config=goal_configs[0][0], data=goal_configs[0][1],
                                      id_num=len(self._goal_labels) - 1)
                # elif new_child.is_valid():
                elif len(approx_configs) > 0:
                    self._non_connected_space.add_approximate(approx_configs[0])
                    # self._computeTemperatures(current_node)

        if self._debug_drawer is not None:
            self._debug_drawer.draw_hierarchy(self._root_node)
        logging.debug('[FreeSpaceProximitySampler::sample] The search led to a dead end. Maybe there is ' \
                      + 'sth in our approximate cache!')
        if self._b_return_approximates:
            return SampleData(self._pick_random_approximate())
        return SampleData(None)

    def debug_draw(self):
        # nodesToUpdate = []
        # nodesToUpdate.extend(self._rootNode.getChildren())
        # while len(nodesToUpdate) > 0:
        # nextNode = nodesToUpdate.pop()
        # nodesToUpdate.extend(nextNode.getChildren())
        if self._debug_drawer is not None:
            self._update_temperatures(self._root_node)
            self._debug_drawer.draw_hierarchy(self._root_node)
