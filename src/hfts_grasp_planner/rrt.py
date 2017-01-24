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
    def __init__(self, config, data=None, dataCopyFct=copy.deepcopy, idNum=-1):
        self._config = config
        self._id = idNum
        self._data = data
        self._dataCopyFct = dataCopyFct

    def getConfiguration(self):
        return self._config

    def getData(self):
        return self._data

    def copy(self):
        copiedData = None
        if self._data is not None:
            copiedData = self._dataCopyFct(self._data)
        return SampleData(numpy.copy(self._config), copiedData, dataCopyFct=self._dataCopyFct, idNum=self._id)

    def isValid(self):
        return self._config is not None

    def isEqual(self, otherSampleData):
        return (self._config == otherSampleData._config).all() and self._data == otherSampleData._data

    def getId(self):
        return self._id

    # def setId(self, idNum):
    #     self._id = idNum

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

    def getSampleData(self):
        return self._data

    def getId(self):
        return self._id

    def getParentId(self):
        return self._parent

    def addChildId(self, cid):
        self._children.append(cid)

    def getChildren(self):
        return self._children

    def __str__(self):
        return "{TreeNode: [id=" + str(self._id) + ", Data=" + str(self._data) + "]}"


class Tree(object):
    TREE_ID = 0
    def __init__(self, rootData, bForwardTree=True):
        self._nodes = [TreeNode(0, 0, rootData.copy())]
        self._labeledNodes = []
        self._node_id = 1
        self._bForwardTree = bForwardTree
        self._tree_id = Tree.TREE_ID + 1
        Tree.TREE_ID = Tree.TREE_ID + 1

    def add(self, parent, childData):
        """
            Adds the given data as a child node of parent.
            @param parent: Must be of type TreeNode and denotes the parent node.
            @paran childData: SampleData that is supposed to be saved in the child node (it is copied).
        """
        childNode = TreeNode(self._node_id, parent.getId(), childData.copy())
        parent.addChildId(childNode.getId())
        self._nodes.append(childNode)
        # self._parents.append(parent.getId())
        # assert(len(self._parents) == self._node_id + 1)
        self._node_id += 1
        return childNode

    def getId(self):
        return self._tree_id

    def addLabeledNode(self, node):
        self._labeledNodes.append(node)

    def getLabeledNodes(self):
        return self._labeledNodes

    def clearLabeledNodes(self):
        self._labeledNodes = []

    def removeLabeledNode(self, node):
        if node in self._labeledNodes:
            self._labeledNodes.remove(node)

    # def clear(self):
    #     self._nodes = []
    #     self._parents = []
    #     self._node_id = 0

    def nearestNeighbor(self, sample):
        pass

    def extractPath(self, goalNode):
        path = [goalNode.getSampleData()]
        currentNode = goalNode
        while (currentNode.getId() != 0):
            currentNode = self._nodes[currentNode.getParentId()]
            path.append(currentNode.getSampleData())
        path.reverse()
        return path

    def getRootNode(self):
        return self._nodes[0]

    def size(self):
        return len(self._nodes)

    def merge(self, mergeNodeA, otherTree, mergeNodeB):
        """
            Merges this tree with the given tree. The connection is established through nodeA and nodeB,
            for which it is assumed that both nodeA and nodeB represent the same configuration.
            In other words, both the parent and all children of nodeB become children of nodeA.
            Labeled nodes of tree B will be added as labeled nodes of tree A.

            Runtime: O(size(otherTree) * num_labeled_nodes(otherTree))

            @param mergeNodeA The node of this tree where to attach otherTree
            @param otherTree  The other tree (is not changed)
            @param mergeNodeB The node of tree B that is merged with mergeNodeA from this tree.

            @return The root of treeB as a TreeNode of treeA after the merge.
        """
        nodeStack = [(mergeNodeA, mergeNodeB, None)]
        bRootNodeInA = None
        while len(nodeStack) > 0:
            (currentNodeA, currentNodeB, ignoreId) = nodeStack.pop()
            for childId in currentNodeB.getChildren():
                if childId == ignoreId:  # prevent adding duplicates
                    continue
                childNodeB = otherTree._nodes[childId]
                childNodeA = self.add(currentNodeA, childNodeB.getSampleData())
                if childNodeB in otherTree._labeledNodes:
                    self.addLabeledNode(childNodeA)
                nodeStack.append((childNodeA, childNodeB, currentNodeB.getId()))

            # In case currentNodeB is not the root of B, we also need to add the parent
            # as a child in this tree.
            parentId = currentNodeB.getParentId()
            if currentNodeB.getId() != parentId:
                if parentId != ignoreId:  # prevent adding duplicates
                    parentNodeB = otherTree._nodes[currentNodeB.getParentId()]
                    childNodeA = self.add(currentNodeA, parentNodeB.getSampleData())
                    nodeStack.append((childNodeA, parentNodeB, currentNodeB.getId()))
                    if parentNodeB in otherTree._labeledNodes:
                        self.addLabeledNode(childNodeA)
            else:  # save the root to return it
                bRootNodeInA = currentNodeA

        return bRootNodeInA


class SqrtTree(Tree):
    def __init__(self, root):
        super(SqrtTree, self).__init__(root)

    def add(self, parent, child):
        childNode = super(SqrtTree, self).add(parent, child)
        self._updateStride()
        return childNode

    # def clear(self):
    #     super(SqrtTree, self).clear()
    #     self.offset = 0
    #     self.stride = 0

    def nearestNeighbor(self, q):
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
                dt = numpy.linalg.norm(q - n.getSampleData().getConfiguration())
                if (dt < d):
                    d = dt
                    nn = n
            self.offset = random.randint(0, self.stride)
        return nn

    def _updateStride(self):
        self.stride = int(1 + math.floor(math.sqrt(len(self._nodes))))


class RTreeTree(Tree):
    def __init__(self, root, dimension, scalingFactors, bForwardTree=True):
        super(RTreeTree, self).__init__(root, bForwardTree=bForwardTree)
        self._scalingFactors = scalingFactors
        self._createIndex(dimension)
        self.dimension = dimension
        self._addToIdx(self._nodes[0])

    def add(self, parent, childData):
        childNode = super(RTreeTree, self).add(parent, childData)
        self._addToIdx(childNode)
        return childNode

    # def clear(self):
    #     super(RTreeTree, self).clear()
    #     self._createIndex(self.dimension)
    #     self._tid = 0

    def nearestNeighbor(self, sampleData):
        if len(self._nodes) == 0:
            return None
        pointList = list(sampleData.getConfiguration())
        pointList = map(lambda x, y: math.sqrt(x) * y, self._scalingFactors, pointList)
        pointList += pointList
        nns = list(self.idx.nearest(pointList))
        return self._nodes[nns[0]]

    def _addToIdx(self, childNode):
        pointList = list(childNode.getSampleData().getConfiguration())
        pointList = map(lambda x, y: math.sqrt(x) * y, self._scalingFactors, pointList)
        pointList += pointList
        self.idx.insert(childNode.getId(), pointList)

    def _createIndex(self, dim):
        prop = index.Property()
        prop.dimension = dim
        self.idx = index.Index(properties=prop)


class Constraint(object):
   def project(self, oldConfig, config):
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
        if tree.getId() in self._constraints_storage:
            self._active_constraints.extend(self._constraints_storage[tree.getId()])

    def reset_constraints(self):
        self._active_constraints = []

    def clear(self):
        self._active_constraints = []
        self._constraints_storage = {}

    def register_new_tree(self, tree):
        if self._new_tree_callback is not None:
            self._constraints_storage[tree.getId()] = self._new_tree_callback(tree)

class PGoalProvider(object):
    def computePGoal(self, numTrees):
        pass

class ConstPGoalProvider(PGoalProvider):
    def __init__(self, pGoal):
        self._pGoal = pGoal

    def computePGoal(self, numTrees):
        logging.debug('[ConstPGoalProvider::computePGoal] Returning constant pGoal')
        if numTrees == 0:
            return 1.0
        return self._pGoal

class DynamicPGoalProvider(PGoalProvider):
    def __init__(self, pMax=0.8, goalW=1.2, pGoalMin=0.01):
        self._pMax = pMax
        self._goalW = goalW
        self._pGoalMin = pGoalMin

    def computePGoal(self, numTrees):
        logging.debug('[DynamicPGoalProvider::computePGoal] Returning dynamic pGoal')
        return self._pMax * math.exp(-self._goalW * numTrees) + self._pGoalMin

class StatsLogger:
    def __init__(self):
        self.numBackwardTrees = 0
        self.numGoalSampled = 0
        self.numValidGoalSampled = 0
        self.numApproxGoalSampled = 0
        self.numAttemptedTreeConnects = 0
        self.numSuccessfulTreeConnects = 0
        self.numGoalNodesSampled = 0
        self.numCFreeSamples = 0
        self.numAccumulatedLogs = 1
        self.finalGraspQuality = 0.0
        self.runtime = 0.0
        self.success = 0
        self.treeSizes = {}

    def clear(self):
        self.numBackwardTrees = 0
        self.numGoalSampled = 0
        self.numValidGoalSampled = 0
        self.numApproxGoalSampled = 0
        self.numAttemptedTreeConnects = 0
        self.numSuccessfulTreeConnects = 0
        self.numGoalNodesSampled = 0
        self.numCFreeSamples = 0
        self.numAccumulatedLogs = 1
        self.finalGraspQuality = 0.0
        self.runtime = 0.0
        self.success = 0
        self.treeSizes = {}

    def toDict(self):
        aDict = {}
        aDict['numBackwardTrees'] = self.numBackwardTrees
        aDict['numGoalSampled'] = self.numGoalSampled
        aDict['numValidGoalSampled'] = self.numValidGoalSampled
        aDict['numApproxGoalSampled'] = self.numApproxGoalSampled
        aDict['numGoalNodesSampled'] = self.numGoalNodesSampled
        aDict['numSuccessfulTreeConnects'] = self.numSuccessfulTreeConnects
        aDict['numCFreeSamples'] = self.numCFreeSamples
        aDict['finalGraspQuality'] = float(self.finalGraspQuality)
        aDict['runtime'] = self.runtime
        aDict['success'] = self.success
        return aDict

    def printLogs(self):
        print 'Logs:'
        print '     numBackwardTrees(avg): ', self.numBackwardTrees
        print '     numGoalSampled(avg): ', self.numGoalSampled
        print '     numValidGoalSampled(avg): ', self.numValidGoalSampled
        print '     numApproxGoalSampled(avg): ', self.numApproxGoalSampled
        print '     numGoalNodesSampled(avg): ', self.numGoalNodesSampled
        print '     numAttemptedTreeConnects(avg): ', self.numAttemptedTreeConnects
        print '     numSuccessfulTreeConnects(avg): ', self.numSuccessfulTreeConnects
        print '     numCFreeSamples(avg): ', self.numCFreeSamples
        print '     finalGraspQuality(avg): ', self.finalGraspQuality
        print '     runtime(avg): ', self.runtime
        print '     success(avg): ', self.success
        if self.numAccumulatedLogs == 1:
            print '     treeSizes: ', self.treeSizes

    def accumulate(self, otherLogger):
        self.numBackwardTrees += otherLogger.numBackwardTrees
        self.numGoalSampled += otherLogger.numGoalSampled
        self.numValidGoalSampled += otherLogger.numValidGoalSampled
        self.numApproxGoalSampled += otherLogger.numApproxGoalSampled
        self.numAttemptedTreeConnects += otherLogger.numAttemptedTreeConnects
        self.numSuccessfulTreeConnects += otherLogger.numSuccessfulTreeConnects
        self.numGoalNodesSampled += otherLogger.numGoalNodesSampled
        self.numCFreeSamples += otherLogger.numCFreeSamples
        self.numAccumulatedLogs += otherLogger.numAccumulatedLogs
        self.finalGraspQuality += otherLogger.finalGraspQuality
        self.runtime += otherLogger.runtime
        self.success += otherLogger.success
        self.treeSizes.update(otherLogger.treeSizes)

    def finalizeAccumulation(self):
        self.numBackwardTrees = float(self.numBackwardTrees) / float(self.numAccumulatedLogs)
        self.numGoalSampled = float(self.numGoalSampled) / float(self.numAccumulatedLogs)
        self.numValidGoalSampled = float(self.numValidGoalSampled) / float(self.numAccumulatedLogs)
        self.numApproxGoalSampled = float(self.numApproxGoalSampled) / float(self.numAccumulatedLogs)
        self.numAttemptedTreeConnects = float(self.numAttemptedTreeConnects) / float(self.numAccumulatedLogs)
        self.numSuccessfulTreeConnects = float(self.numSuccessfulTreeConnects) / float(self.numAccumulatedLogs)
        self.numGoalNodesSampled = float(self.numGoalNodesSampled) / float(self.numAccumulatedLogs)
        self.finalGraspQuality = self.finalGraspQuality / float(self.numAccumulatedLogs)
        self.runtime = self.runtime / float(self.numAccumulatedLogs)
        self.success = self.success / float(self.numAccumulatedLogs)
        self.numCFreeSamples = float(self.numCFreeSamples) / float(self.numAccumulatedLogs)


class RRT:
    def __init__(self, pGoalProvider, cfreeSampler, goalSampler, logger, pGoalTree=0.8, constraintsManager=None):  # pForwardTree, pConnectTree
        """ Initializes the RRT planner
            @param pGoal - Instance of PGoalProvider that provides a probability of sampling a new goal
            @param cfreeSampler - A sampler of c_free.
            @param goalSampler - A sampler of the goal region.
            @param logger - A logger (of type Logger) for printouts.
            @param constraintsManager - (optional) a constraint manager.
        """
        self.pGoalProvider = pGoalProvider
        self.pGoalTree = pGoalTree
        self.goalSampler = goalSampler
        self.cfreeSampler = cfreeSampler
        self.logger = logger
        self.statsLogger = StatsLogger()
        # self.debugConfigList = []
        if constraintsManager is None:
            constraintsManager = ConstraintsManager()
        self._constraintsManager = constraintsManager

    def extend(self, tree, randomSample, addIntermediates=True, addTreeStep=10):
        self._constraintsManager.set_active_tree(tree)
        nearestNode = tree.nearestNeighbor(randomSample)
        (bConnected, samples) = self.cfreeSampler.interpolate(nearestNode.getSampleData(), randomSample,
                                                              projectionFunction=self._constraintsManager.project)
        parentNode = nearestNode
        self.logger.debug('[RRT::extend We have ' + str(len(samples)-1) + " intermediate configurations")
        if addIntermediates:
            for i in range(addTreeStep, len(samples)-1, addTreeStep):
                parentNode = tree.add(parentNode, samples[i].copy())
        if len(samples) > 1:
            lastNode = tree.add(parentNode, samples[-1].copy())
        else:
            # self.debugConfigList.extend(samples)
            lastNode = parentNode
        return (lastNode, bConnected)

    def rrtHierarchy(self, startConfig, timeLimit=60, debugFunction=lambda x: x):
        """ RRT algorithm with approximate goal region """
        self.goalSampler.clear()
        self.statsLogger.clear()
        startSample = SampleData(startConfig)
        tree = RTreeTree(startSample, self.cfreeSampler.getSpaceDimension())
        solutionFound = False
        startTime = time.clock()
        goalNode = None
        debugFunction(tree)
        while time.clock() < startTime + timeLimit and not solutionFound:
            p = random.random()
            if (p > self.pGoal):
                randomSample = self.cfreeSampler.sample()
            else:
                randomSample = self.goalSampler.sample()
            if not randomSample.isValid():
                continue
            (newNode, bConnected) = self.extend(tree, randomSample)

            # Check whether we reached our approximate goal
            if (self.goalSampler.isApproxGoal(newNode.getSampleData())):
                # if so, let's see whether this is a real goal
                if (self.goalSampler.is_goal(newNode.getSampleData())):
                    solutionFound = True
                    goalNode = newNode
                else:
                    # tell the sampler to refine the goal sampling in this area
                    self.goalSampler.refineGoalSampling(newNode.getSampleData())
            debugFunction(tree)

        if solutionFound:
            self.logger.info('Found a solution!')
            return tree.extractPath(goalNode)
        return None

    def _growTrees(self, treeA, treeB, randomSample):
        (newNodeA, bConnected) = self.extend(treeA, randomSample)
        (newNodeB, bConnected) = self.extend(treeB, newNodeA.getSampleData())
        return (newNodeA, newNodeB, bConnected)

    def _pickRandomBackwardTree(self, goalTrees, approxGoalTrees):
        dice = random.randint(0, 1)
        backwardTree = None
        bGoalTree = False
        if dice == 0 and len(goalTrees) > 0:
            backwardTree = random.choice(goalTrees)
            bGoalTree = True
        elif len(approxGoalTrees) > 0:
            backwardTree = random.choice(approxGoalTrees)
        return (backwardTree, bGoalTree)

    def _biRRT_helper_nearestTree(self, sample, backwardTrees):
        nn = None
        dist = float('inf')
        tree = None
        for treeTemp in backwardTrees:
            nnTemp = treeTemp.nearestNeighbor(sample)
            distTemp = self.cfreeSampler.distance(sample.getConfiguration(),
                                                  nnTemp.getSampleData().getConfiguration())
            if distTemp < dist:
                dist = distTemp
                nn = nnTemp
                tree = treeTemp
        return (tree, nn)

    def biRRT(self, startConfig, useHierarchy=True, timeLimit=60, debugFunction=lambda x, y: None,
              shortcutTime=5.0, timerFunction=time.time):
        """ Bidirectional RRT algorithm with approximate goal region. """
        self.goalSampler.clear()
        self.statsLogger.clear()
        self._constraintsManager.clear()
        forwardTree = RTreeTree(SampleData(startConfig), self.cfreeSampler.getSpaceDimension(),
                               self.cfreeSampler.getScalingFactors())
        self._constraintsManager.register_new_tree(forwardTree)
        backwardTrees = []
        pathFound = False
        path = None
        startTime = timerFunction()
        debugFunction(forwardTree, backwardTrees)
        bSearchingForward = True
        # self.debugConfigList = []

        # Main loop
        self.logger.debug('[RRT::biRRT] Starting planning loop')
        while timerFunction() < startTime + timeLimit and not pathFound:
            debugFunction(forwardTree, backwardTrees)
            p = random.random()
            self.logger.debug('[RRT::biRRT] Rolled a dice: ' + str(p))
            pGoal = self.pGoalProvider.computePGoal(len(backwardTrees))
            if p <= pGoal:
                # Create a new backward tree
                self.logger.debug('[RRT::biRRT] Creating a new backward tree')
                goalSample = self.goalSampler.sample(False)
                self.statsLogger.numGoalSampled += 1
                self.logger.debug('[RRT::biRRT] Sampled a new goal: ' + str(goalSample))
                if goalSample.isValid():
                    self.statsLogger.numValidGoalSampled += 1
                    bTree = RTreeTree(goalSample, self.cfreeSampler.getSpaceDimension(),
                                      self.cfreeSampler.getScalingFactors(), bForwardTree=False)
                    backwardTrees.append(bTree)
                    self._constraintsManager.register_new_tree(bTree)
                    self.statsLogger.numBackwardTrees += 1
                    self.logger.debug('[RRT::biRRT] Goal is valid; created new backward tree')
            else:
                # Extend search trees
                self.logger.debug('[RRT::biRRT] Extending search trees')
                self._constraintsManager.reset_constraints()
                randomSample = self.cfreeSampler.sample()
                self.logger.debug('[RRT::biRRT] Drew random sample: ' + str(randomSample))
                self.statsLogger.numCFreeSamples += 1
                (forwardNode, backwardNode, backwardTree, bConnected) = (None, None, None, False)
                if bSearchingForward or len(backwardTrees) == 0:
                    self.logger.debug('[RRT::biRRT] Extending forward tree to random sample')
                    (forwardNode, bConnected) = self.extend(forwardTree, randomSample)
                    self.logger.debug('[RRT::biRRT] Forward tree connected to sample: ' + str(bConnected))
                    self.logger.debug('[RRT::biRRT] New forward tree node: ' + str(forwardNode))
                    self.logger.debug('[RRT::biRRT] Attempting to connect forward tree to backward tree')
                    if len(backwardTrees) > 0:
                        (backwardTree, nearestNode) = self._biRRT_helper_nearestTree(forwardNode.getSampleData(), backwardTrees)
                        (backwardNode, bConnected) = self.extend(backwardTree, forwardNode.getSampleData())
                    else:
                        bConnected = False
                else:
                    #(backwardTree, nearestNode) = self._biRRT_helper_nearestTree(randomSample, backwardTrees)
                    self.logger.debug('[RRT::biRRT] Extending backward tree to random sample')
                    backwardTree = random.choice(backwardTrees)
                    (backwardNode, bConnected) = self.extend(backwardTree, randomSample)
                    self.logger.debug('[RRT::biRRT] New backward tree node: ' + str(backwardNode))
                    self.logger.debug('[RRT::biRRT] Backward tree connected to sample: ' + str(bConnected))
                    self.logger.debug('[RRT::biRRT] Attempting to connect forward tree to backward tree')
                    (forwardNode, bConnected) = self.extend(forwardTree, backwardNode.getSampleData())
                self.statsLogger.numAttemptedTreeConnects += 1
                if bConnected:
                    self.logger.debug('[RRT::biRRT] Trees connected')
                    self.statsLogger.numSuccessfulTreeConnects += 1
                    treeName = 'merged_backward_tree' + str(self.statsLogger.numSuccessfulTreeConnects)
                    self.statsLogger.treeSizes[treeName] = backwardTree.size()
                    rootB = forwardTree.merge(forwardNode, backwardTree, backwardNode)
                    backwardTrees.remove(backwardTree)
                    self._constraintsManager.reset_constraints()
                    if useHierarchy:
                        forwardTree.addLabeledNode(rootB)
                        (pathFound, path, newBackwardTree) = self._biRRT_helper_checkForGoal(forwardTree)
                        if newBackwardTree is not None:
                            backwardTrees.append(newBackwardTree)
                            self._constraintsManager.register_new_tree(newBackwardTree)
                            self.statsLogger.numBackwardTrees += 1
                            self.logger.debug('[RRT:biRRT] Received a new backward tree from checkForGoal')
                    else:
                        path = forwardTree.extractPath(rootB)
                        pathFound = True
                bSearchingForward = not bSearchingForward

        self.statsLogger.treeSizes['forward_tree'] = forwardTree.size()
        for i in range(len(backwardTrees)):
            self.statsLogger.treeSizes['unmerged_backward_tree' + str(i)] = backwardTrees[i].size()
        debugFunction(forwardTree, backwardTrees)
        self.statsLogger.numGoalNodesSampled = self.goalSampler.getNumGoalNodesSampled()
        self.statsLogger.runtime = timerFunction() - startTime
        if path is not None:
            self.statsLogger.finalGraspQuality = self.goalSampler.getQuality(path[-1])
            self.statsLogger.success = 1
        # if useHierarchy and path is not None:
        #     self.statsLogger.connectedGoalFromHotRegion = self.goalSampler.goalIsFromHotRegion(path[-1])
        return self.shortcut(path, shortcutTime)

    def proximityBiRRT(self, startConfig, timeLimit=60, debugFunction=lambda x, y: None,
                       shortcutTime=5.0, timerFunction=time.time):
        """ Bidirectional RRT algorithm with hierarchical goal region that
            uses free space proximity to bias sampling. """
        if not self.cfreeSampler.isValid(startConfig):
            self.logger.info('[RRT::proximityBiRRT] Start configuration is invalid. Aborting.')
            return None
        from sampler import FreeSpaceProximitySampler, FreeSpaceModel, ExtendedFreeSpaceModel
        assert type(self.goalSampler) == FreeSpaceProximitySampler
        self.goalSampler.clear()
        self.statsLogger.clear()
        self._constraintsManager.clear()
        # Create free space memories that our goal sampler needs
        connectedFreeSpace = FreeSpaceModel(self.cfreeSampler)
        nonConnectedFreeSpace = ExtendedFreeSpaceModel(self.cfreeSampler)
        self.goalSampler.setConnectedSpace(connectedFreeSpace)
        self.goalSampler.setNonConnectedSpace(nonConnectedFreeSpace)
        # Create forward tree
        forwardTree = RTreeTree(SampleData(startConfig), self.cfreeSampler.getSpaceDimension(),
                               self.cfreeSampler.getScalingFactors())
        self._constraintsManager.register_new_tree(forwardTree)
        connectedFreeSpace.addTree(forwardTree)
        # Various variable initializations
        backwardTrees = []
        goalTreeIds = []
        pathFound = False
        path = None
        bSearchingForward = True
        # self.debugConfigList = []
        # Start
        startTime = timerFunction()
        debugFunction(forwardTree, backwardTrees)

        # Main loop
        self.logger.debug('[RRT::proximityBiRRT] Starting planning loop')
        while timerFunction() < startTime + timeLimit and not pathFound:
            debugFunction(forwardTree, backwardTrees)
            p = random.random()
            pGoal = self.pGoalProvider.computePGoal(len(backwardTrees))
            self.logger.debug('[RRT::proximityBiRRT] Rolled a dice: ' + str(p) + '. pGoal is ' +
                              str(pGoal))
            if p < pGoal:
            # if p < self.pGoal or len(backwardTrees) == 0:
                # Create a new backward tree
                self.logger.debug('[RRT::proximityBiRRT] Sampling a new goal configuration')
                goalSample = self.goalSampler.sample()
                self.statsLogger.numGoalSampled += 1
                self.logger.debug('[RRT::proximityBiRRT] Sampled a new goal: ' + str(goalSample))
                if goalSample.isValid():
                    bTree = RTreeTree(goalSample, self.cfreeSampler.getSpaceDimension(),
                                      self.cfreeSampler.getScalingFactors(), bForwardTree=False)
                    self._constraintsManager.register_new_tree(bTree)
                    if self.goalSampler.is_goal(goalSample):
                        self.statsLogger.numValidGoalSampled += 1
                        self.logger.debug('[RRT::proximityBiRRT] Goal sample is valid.' \
                                          + ' Created new backward tree')
                        goalTreeIds.append(bTree.getId())
                    else:
                        self.statsLogger.numApproxGoalSampled += 1
                        self.logger.debug('[RRT::proximityBiRRT] Goal sample is valid, but approximate.' \
                                          + ' Created new approximate backward tree')
                    self.statsLogger.numBackwardTrees += 1
                    backwardTrees.append(bTree)
                    nonConnectedFreeSpace.addTree(bTree)
            else:
                # Extend search trees
                self.logger.debug('[RRT::proximityBiRRT] Extending search trees')
                self._constraintsManager.reset_constraints()
                randomSample = self.cfreeSampler.sample()
                self.logger.debug('[RRT::proximityBiRRT] Drew random sample: ' + str(randomSample))
                self.statsLogger.numCFreeSamples += 1
                (forwardNode, backwardNode, backwardTree, bConnected) = (None, None, None, False)
                if bSearchingForward or len(backwardTrees) == 0:
                    self.logger.debug('[RRT::proximityBiRRT] Extending forward tree to random sample')
                    (forwardNode, bConnected) = self.extend(forwardTree, randomSample)
                    self.logger.debug('[RRT::proximityBiRRT] Forward tree connected to sample: ' + str(bConnected))
                    self.logger.debug('[RRT::proximityBiRRT] New forward tree node: ' + str(forwardNode))
                    if len(backwardTrees) > 0:
                        self.logger.debug('[RRT::proximityBiRRT] Attempting to connect forward tree ' \
                                          + 'to backward tree')
                        (backwardTree, nearestNode) = \
                            self._biRRT_helper_nearestTree(forwardNode.getSampleData(), backwardTrees)
                        (backwardNode, bConnected) = self.extend(backwardTree, forwardNode.getSampleData())
                    else:
                        bConnected = False
                else:
                    self.logger.debug('[RRT::proximityBiRRT] Extending backward tree to random sample')
                    # TODO try closest tree instead
                    backwardTree = self._proximityBiRRT_helper_pickBackwardTree(backwardTrees,
                                                                                goalTreeIds)
                    # (backwardTree, nearestNode) = self._biRRT_helper_nearestTree(randomSample, backwardTrees)
                    if backwardTree.getId() in goalTreeIds:
                        self.logger.debug('[RRT::proximityBiRRT] Attempting to connect goal tree!!!!')
                    (backwardNode, bConnected) = self.extend(backwardTree, randomSample)
                    self.logger.debug('[RRT::proximityBiRRT] New backward tree node: ' + str(backwardNode))
                    self.logger.debug('[RRT::proximityBiRRT] Backward tree connected to sample: ' \
                                      + str(bConnected))
                    self.logger.debug('[RRT::proximityBiRRT] Attempting to connect forward tree ' \
                                      + 'to backward tree ' + str(backwardTree.getId()))
                    (forwardNode, bConnected) = self.extend(forwardTree, backwardNode.getSampleData())
                self.statsLogger.numAttemptedTreeConnects += 1
                if bConnected:
                    self.logger.debug('[RRT::proximityBiRRT] Trees connected')
                    self.statsLogger.numSuccessfulTreeConnects += 1
                    treeName = 'merged_backward_tree' + str(self.statsLogger.numSuccessfulTreeConnects)
                    self.statsLogger.treeSizes[treeName] = backwardTree.size()
                    rootB = forwardTree.merge(forwardNode, backwardTree, backwardNode)
                    backwardTrees.remove(backwardTree)
                    nonConnectedFreeSpace.removeTree(backwardTree.getId())
                    # Check whether we connected to a goal tree or not
                    if backwardTree.getId() in goalTreeIds:
                        goalTreeIds.remove(backwardTree.getId())
                        path = forwardTree.extractPath(rootB)
                        pathFound = True
                        self.logger.debug('[RRT::proximityBiRRT] Found a path!')
                bSearchingForward = not bSearchingForward

        self.statsLogger.treeSizes['forward_tree'] = forwardTree.size()
        for bwTree in backwardTrees:
            self.statsLogger.treeSizes['unmerged_backward_tree' + str(bwTree.getId())] = bwTree.size()
        debugFunction(forwardTree, backwardTrees)
        self.goalSampler.debugDraw()
        self.statsLogger.numGoalNodesSampled = self.goalSampler.getNumGoalNodesSampled()
        self.statsLogger.runtime = timerFunction() - startTime
        if path is not None:
            self.statsLogger.finalGraspQuality = self.goalSampler.getQuality(path[-1])
            self.statsLogger.success = 1
        return self.shortcut(path, shortcutTime)

    def _proximityBiRRT_helper_pickBackwardTree(self, backwardTrees, goalTreeIds):
        p = random.random()
        goalTrees = [x for x in backwardTrees if x.getId() in goalTreeIds]
        nonGoalTrees = [x for x in backwardTrees if x.getId() not in goalTreeIds]
        if p < self.pGoalTree and len(goalTreeIds) > 0:
            return random.choice(goalTrees)
        elif len(nonGoalTrees) > 0:
            return random.choice(nonGoalTrees)
        elif len(backwardTrees) > 0: # this may happen if we have only goal trees, but p >= self.pGoalTree
            return random.choice(backwardTrees)
        else:
            raise ValueError('We do not have any backward trees to pick from')

#     def _biRRT_helper_sampleGoal(self, backwardTrees):
        # goalSample = self.goalSampler.sample(False)
        # # print 'sampled a new goal....'
        # if not goalSample.isValid():
            # return
        # if len(backwardTrees) > 0:
            # bTree = random.choice(backwardTrees)
            # (newNode, bConnected) = self.extend(bTree, goalSample)
            # if bConnected:
                # bTree.addLabeledNode(newNode)
                # # print '...added it to existing tree'
                # return
            # # print '...only extended towards it'

        # bTree = RTreeTree(goalSample, self.cfreeSampler.getSpaceDimension())
        # backwardTrees.append(bTree)
        # # print '...created new tree ', len(backwardTrees)

    # def _biRRT_helper_growTree(self, tree, sample=None):
        # if sample is None:
            # sample = self.cfreeSampler.sample()
        # if sample.isValid():
            # self.extend(tree, sample)

#     def _biRRT_helper_connectTrees(self, treeA, treeB):
        # randomSample = self.cfreeSampler.sample()
        # (newNode, bConnected) = self.extend(treeA, randomSample)
        # (newNodeB, bConnected) = self.extend(treeB, newNode.getSampleData())
        # if bConnected:
            # rootB = treeA.merge(newNode, treeB, newNodeB)
            # treeA.addLabeledNode(rootB)
        # return bConnected

    def _biRRT_helper_greedilyRefineHotRegion(self, goalCandidate, tree):
        self.logger.debug('[RRT:_biRRT_helper_greedilyRefineHotRegion] refining hot region...')
        goalSample = self.goalSampler.refineAndSample(goalCandidate)
        bConnected = True
        while goalSample.isValid() and bConnected:
            self.logger.debug('[RRT:_biRRT_helper_greedilyRefineHotRegion] Attempting to connect to new goal...')
            (newNode, bConnected) = self.extend(tree, goalSample)
            if self.goalSampler.is_goal(goalSample) and bConnected:
                # print '...we connected to a goal'
                self.logger.debug('[RRT:_biRRT_helper_greedilyRefineHotRegion] Connected to a final goal!')
                return (True, tree.extractPath(newNode), None)
            elif bConnected:
                self.logger.debug('[RRT:_biRRT_helper_greedilyRefineHotRegion] Connected to approx goal!')
                goalSample = self.goalSampler.refineAndSample(goalSample)
            else:
                self.logger.debug('[RRT:_biRRT_helper_greedilyRefineHotRegion] Failed to connect, creating' \
                                  + ' new tree')
                self.statsLogger.numGreedyRefinesFailed += 1
                return (False, None, \
                        RTreeTree(goalSample, self.cfreeSampler.getSpaceDimension(), bForwardTree=False))
        self.logger.debug('[RRT:_biRRT_helper_greedilyRefineHotRegion] Last goal sample was invalid, failed!')
        self.statsLogger.numGreedyRefinesFailed += 1
        return (False, None, None)

    def _biRRT_helper_checkForGoal(self, tree):
        # print 'checking whether we connected to a goal'
        for lnode in tree.getLabeledNodes():
            goalCandidate = lnode.getSampleData()
            if self.goalSampler.is_goal(goalCandidate):
                # print 'Found a goal!!!!'
                return (True, tree.extractPath(lnode), None)
            else:
                self.statsLogger.numGreedyRefines += 1
                (pathFound, path, bTree) = self._biRRT_helper_greedilyRefineHotRegion(goalCandidate, tree)
                if pathFound:
                    return (pathFound, path, bTree)
                # self.goalSampler.refineGoalSampling(goalCandidate)
        tree.clearLabeledNodes()

        return (False, None, None)

    def shortcut(self, path, timeLimit):
        if path is None:
            return None
        self.logger.debug('[RRT::shortcut] Shortcutting path of length %i with timelimit %f' % (len(path),
                                                                                                timeLimit))
        startTime = time.clock()
        allPairs = [(i,j) for i in range(len(path)) for j in range(i+2, len(path))]
        random.shuffle(allPairs)
        while time.clock() < startTime + timeLimit and len(allPairs) > 0:
            indexPair = allPairs.pop()
            (bSuccess, samples) = self.cfreeSampler.interpolate(path[indexPair[0]], path[indexPair[1]])
            if bSuccess:
                path[indexPair[0] + 1:] = path[indexPair[1]:]
                allPairs = [(i,j) for i in range(len(path)) for j in range(i+2, len(path))]
                random.shuffle(allPairs)
        self.logger.debug('[RRT::shortcut] Shorcutting finished. New path length %i' % len(path))
        return path

    def classicalRRT(self, startConfig, timeLimit=60, debugFunction=lambda x: x):
        """ Classical RRT algorithm """
        self.goalSampler.clear()
        startSample = SampleData(startConfig)
        tree = RTreeTree(startSample, self.cfreeSampler.getSpaceDimension())
        solutionFound = False
        startTime = time.clock()
        goalNode = None
        debugFunction(tree)
        while time.clock() < startTime + timeLimit and not solutionFound:
            p = random.random()
            if (p > self.pGoal):
                randomSample = self.cfreeSampler.sample()
            else:
                randomSample = self.goalSampler.sample()
            if not randomSample.isValid():
                continue
            (newNode, bConnected) = self.extend(tree, randomSample)

            if (self.goalSampler.is_goal(newNode.getSampleData())):
                solutionFound = True
                goalNode = newNode
            debugFunction(tree)

        if solutionFound:
            self.logger.info('Found a solution!')
            return tree.extractPath(goalNode)
        return None

    def classicalBiRRT(self, startConfig, timeLimit=60, debugFunction=lambda x, y: None):
        """ Bidirectional RRT algorithm. """
        return self.biRRT(startConfig=startConfig, timeLimit=timeLimit, useHierarchy=False, debugFunction=debugFunction)

