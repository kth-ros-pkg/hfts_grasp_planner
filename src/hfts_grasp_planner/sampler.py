#!/usr/bin/env python
""" This module contains a general hierarchically organized goal region sampler. """

import ast
import logging
import math
import numpy
import random
from blist import sortedlist
from rtree import index
from rrt import SampleData

NUMERICAL_EPSILON = 0.00001

class SamplingResult:
    def __init__(self, configuration, hierarchyInfo=None, dataExtractor=None, cacheId=-1, bOriginatesFromHotRegion=False):
        self.configuration = configuration
        self.dataExtractor = dataExtractor
        self.hierarchyInfo = hierarchyInfo
        self.cacheId = cacheId
        self.bOriginatesFromHotRegion = bOriginatesFromHotRegion

    def getConfiguration(self):
        return self.configuration

    def toSampleData(self):
        if self.dataExtractor is not None:
            return SampleData(self.configuration, self.dataExtractor.extractData(self.hierarchyInfo),
                              self.dataExtractor.getCopyFunction(), idNum=self.cacheId)
        return SampleData(self.configuration, idNum=self.cacheId)

    def isValid(self):
        return self.hierarchyInfo.isValid()

    def isGoal(self):
        if self.hierarchyInfo is not None:
            return self.hierarchyInfo.is_goal()
        return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{SamplingResult:[Config=" + str(self.configuration) + "; Info=" + str(self.hierarchyInfo) + "]}"

class CSpaceSampler:
    def sample(self):
        pass

    def sampleGaussianNeighborhood(self, config, variance):
        pass

    def isValid(self, qSample):
        pass

    def getSamplingStep(self):
        pass

    def getSpaceDimension(self):
        pass

    def getUpperBounds(self):
        pass

    def getLowerBounds(self):
        pass

    def getScalingFactors(self):
        return self.getSpaceDimension() * [1]

    def distance(self, configA, configB):
        totalSum = 0.0
        scalingFactors = self.getScalingFactors()
        for i in range(len(configA)):
            totalSum += scalingFactors[i] * math.pow(configA[i] - configB[i], 2)
        return math.sqrt(totalSum)

    def configsAreEqual(self, configA, configB):
        dist = self.distance(configA, configB)
        return dist < NUMERICAL_EPSILON

    def interpolate(self, startSample, endSample, projectionFunction=lambda x,y: y):
        """
        Samples cspace linearly from the startSample to endSample until either
        a collision is detected or endSample is reached. All intermediate sampled configurations
        are returned in a list as SampleData.
        If a projectionFunction is specified, each sampled configuration is projected using this
        projection function. This allows to interpolate within a constraint manifold, i.e. some subspace of
        the configuration space. Additionally to the criterias above, the method also terminates when
        no more progress is made towards endSample.
        @param startSample        The SampleData to start from.
        @param endSample          The SampleData to sample to.
        @param projectionFunction (Optional) A projection function on a contraint manifold.
        @return A tuple (bSuccess, samples), where bSuccess is True if a connection to endSample was found;
                samples is a list of all intermediate sampled configurations [startSample, ..., lastSampled].
        """
        waypoints = [startSample]
        configSample = startSample.getConfiguration()
        preConfigSample = startSample.getConfiguration()
        distToTarget = self.distance(endSample.getConfiguration(), configSample)
        while True:
            preDistToTarget = distToTarget
            distToTarget = self.distance(endSample.getConfiguration(), configSample)
            if self.configsAreEqual(configSample, endSample.getConfiguration()):
                # We reached our target. Since we want to keep data stored in the target, simply replace
                # the last waypoint with the instance endSample
                waypoints.pop()
                waypoints.append(endSample)
                return (True, waypoints)
            elif distToTarget > preDistToTarget:
                # The last sample we added, took us further away from the target, then the previous one.
                # Hence remove the last sample and return.
                waypoints.pop()
                return (False, waypoints)
            # We are still making progress, so sample a new sample
            # To prevent numerical issues, we move at least NUMERICAL_EPSILON
            step = min(self.getSamplingStep(), max(distToTarget, NUMERICAL_EPSILON))
            configSample = configSample + step * (endSample.getConfiguration() - configSample) / distToTarget
            # Project the sample to the constraint manifold
            configSample = projectionFunction(preConfigSample, configSample)
            if configSample is not None and self.isValid(configSample):
                # We have a new valid sample, so add it to the waypoints list
                waypoints.append(SampleData(numpy.copy(configSample)))
                preConfigSample = numpy.copy(configSample)
            else:
                # We ran into an obstacle - we won t get further, so just return what we have so far
                return (False, waypoints)

    def linearPath(self, startSample, endSample):
        """
        Samples cspace linearly from the startSample to endSample until either
        a collision is detected or endSample is reached. All intermediate sampled configurations
        are returned in a list as SampleData.
        @param startSample        The SampleData to start from.
        @param endSample          The SampleData to sample to.
        @return A tuple (bSuccess, samples), where bSuccess is True samples is a list of all
                intermediate sampled configurations [startSample, ..., lastSampled],
            where lastSampled = endSample in case the path .
        """
        cDir = endSample.getConfiguration() - startSample.getConfiguration()
        distance = numpy.linalg.norm(cDir)
        if distance == 0.0:
            print 'Warning: Sampling a path with zero length.'
            return (True, [startSample, endSample])
        cDir = 1.0 / distance * cDir
        nextDistLeft = max(distance, 0.0)
        clastValid = startSample.getConfiguration()
        bConnected = False
        samples = [startSample]

        # First sample outside of the loop
        step = min(self.getSamplingStep(), max(nextDistLeft, 0.00001))
        cnext = startSample.getConfiguration() + step * cDir
        nextDistLeft -= step
        cNextIsValid = self.isValid(cnext)

        # from now on we can loop
        while nextDistLeft > 0.0 and cNextIsValid:
            clastValid = cnext
            samples.append(SampleData(numpy.copy(clastValid)))
            step = min(self.getSamplingStep(), max(nextDistLeft, 0.00001))
            cnext = cnext + step * cDir
            nextDistLeft -= step
            cNextIsValid = self.isValid(cnext)

        if cNextIsValid and nextDistLeft <= 0.0:
            samples.append(endSample)
            bConnected = True

        return (bConnected, samples)

class SimpleHierarchyNode:
    class DummyHierarchyInfo:
        def __init__(self, uniqueLabel):
            self.uniqueLabel = uniqueLabel

        def getUniqueLabel(self):
            return self.uniqueLabel

        def isGoal(self):
            return False

        def isValid(self):
            return False

    def __init__(self, config, hierarchyInfo):
        self.config = config
        self.hierarchyInfo = hierarchyInfo
        self.children = []

    def getChildren(self):
        return self.children

    def getActiveChildren(self):
        return self.children

    def getNumChildren(self):
        return len(self.children)

    def getMaxNumChildren(self):
        return 1 #self.hierarchyInfo.getPossibleNumChildren()

    def getNumLeavesInBranch(self):
        return 0

    def getMaxNumLeavesInBranch(self):
        return 1 #self.hierarchyInfo.getPossibleNumLeaves()

    def getUniqueLabel(self):
        return self.hierarchyInfo.getUniqueLabel()

    def getT(self):
        if self.hierarchyInfo.is_goal() and self.hierarchyInfo.isValid():
            return 1.5
        if self.hierarchyInfo.isValid():
            return 1.0
        return 0.0

    def isGoal(self):
        return self.hierarchyInfo.is_goal()

    def getActiveConfiguration(self):
        return self.config

    def addChild(self, child):
        self.children.append(child)

class NaiveGoalSampler:
    def __init__(self, goalRegion, numIterations=40, debugDrawer=None):
        self.goalRegion = goalRegion
        self.depthLimit = goalRegion.get_max_depth()
        self.goalRegion.set_max_iter(numIterations)
        self._debugDrawer = debugDrawer
        self.clear()

    def clear(self):
        self.cache = []
        self._rootNode = SimpleHierarchyNode(None, self.goalRegion.get_root())
        self._labelNodeMap = {}
        self._labelNodeMap['root'] = self._rootNode
        if self._debugDrawer is not None:
            self._debugDrawer.clear()

    def getNumGoalNodesSampled(self):
        return len(self._labelNodeMap)

    def _computeAncestorLabels(self, uniqueLabel):
        labelAsList = ast.literal_eval(uniqueLabel)
        depth = len(labelAsList) / 3
        nDepth = depth - 1
        ancestorLabels = []
        while nDepth >= 1:
            ancestorLabel = []
            for f in range(3):
                ancestorLabel.extend(labelAsList[f * depth:f * depth + nDepth])
            ancestorLabels.append(str(ancestorLabel))
            labelAsList = ancestorLabel
            depth -= 1
            nDepth = depth - 1
        ancestorLabels.reverse()
        return ancestorLabels

    def _addNewSample(self, sample):
        uniqueLabel = sample.hierarchyInfo.getUniqueLabel()
        ancestorLabels = self._computeAncestorLabels(uniqueLabel)
        parent = self._rootNode
        for ancestorLabel in ancestorLabels:
            if ancestorLabel in self._labelNodeMap:
                parent = self._labelNodeMap[ancestorLabel]
            else:
                ancestorNode = SimpleHierarchyNode(config=None,
                                                   hierarchyInfo=SimpleHierarchyNode.DummyHierarchyInfo(ancestorLabel))
                parent.addChild(ancestorNode)
                self._labelNodeMap[ancestorLabel] = ancestorNode
                parent = ancestorNode
        if uniqueLabel in self._labelNodeMap:
            return
        newNode = SimpleHierarchyNode(config=sample.getConfiguration(), hierarchyInfo=sample.hierarchyInfo)
        self._labelNodeMap[uniqueLabel] = newNode
        parent.addChild(newNode)

    def sample(self, bDummy=False):
        logging.debug('[NaiveGoalSampler::sample] Sampling a goal in the naive way')
        mySample = self.goalRegion.sample(self.depthLimit)
        self._addNewSample(mySample)
        if self._debugDrawer is not None:
            self._debugDrawer.drawHierarchy(self._rootNode)
        if not mySample.isValid() or not mySample.is_goal():
            logging.debug('[NaiveGoalSampler::sample] Failed. Did not get a valid goal!')
            return SampleData(None)
        else:
            mySample.cacheId = len(self.cache)
            self.cache.append(mySample)

        logging.debug('[NaiveGoalSampler::sample] Success. Found a valid goal!')
        return mySample.toSampleData()

    def getQuality(self, sampleData):
        idx = sampleData.getId()
        return self.cache[idx].hierarchyInfo.getQuality()

    def isGoal(self, sample):
        sampledBefore = sample.getId() > 0 and sample.getId() < len(self.cache)
        if sampledBefore:
            return self.goalRegion.is_goal(self.cache[sample.getId()])
        return False
            # return True
        # return self.goalRegion.isGoalConfiguration(qReachable)

class FreeSpaceModel(object):
    def __init__(self, cspaceSampler):
        self._trees = []
        self._cspaceSampler = cspaceSampler

    def addTree(self, tree):
        self._trees.append(tree)

    def removeTree(self, treeId):
        newTreeList = []
        for tree in self._trees:
            if tree.getId() != treeId:
                newTreeList.append(tree)
        self._trees = newTreeList

    def getNearestConfiguration(self, config):
        (dist, nearestConfig) = (float('inf'), None)
        for tree in self._trees:
            treeNode = tree.nearestNeighbor(SampleData(config))
            tmpConfig = treeNode.getSampleData().getConfiguration()
            tmpDist = self._cspaceSampler.distance(config, tmpConfig)
            if tmpDist < dist:
                dist = tmpDist
                nearestConfig = tmpConfig
        return (dist, nearestConfig)


class ExtendedFreeSpaceModel(FreeSpaceModel):
    def __init__(self, cspaceSampler):
        super(ExtendedFreeSpaceModel, self).__init__(cspaceSampler)
        self._scalingFactors = cspaceSampler.getScalingFactors()
        prop = index.Property()
        prop.dimension = cspaceSampler.getSpaceDimension()
        self._approximateIndex = index.Index(properties=prop)
        self._approximateConfigs = []
        self._temporalMiniCache = []

    def getNearestConfiguration(self, config):
        (treeDist, nearestTreeConfig) = super(ExtendedFreeSpaceModel, self).getNearestConfiguration(config)
        (tempDist, tempConfig) = self.getClosestTemporary(config)
        if tempDist < treeDist:
            treeDist = tempDist
            nearestTreeConfig = tempConfig

        nearestApproximate = None
        if len(self._approximateConfigs) > 0:
            pointList = self._makeCoordinates(config)
            nns = list(self._approximateIndex.nearest(pointList))
            nnId = nns[0]
            nearestApproximate = self._approximateConfigs[nnId]
            assert nearestApproximate is not None
            dist = self._cspaceSampler.distance(nearestApproximate, config)
        if nearestTreeConfig is not None and nearestApproximate is not None:
            if dist < treeDist:
                return (dist, nearestApproximate)
            else:
                return (treeDist, nearestTreeConfig)
        elif nearestTreeConfig is not None:
            return (treeDist, nearestTreeConfig)
        elif nearestApproximate is not None:
            return (dist, nearestApproximate)
        else:
            return (float('inf'), None)

    def addTemporary(self, configs):
        self._temporalMiniCache.extend(configs)

    def clearTemporaryCache(self):
        self._temporalMiniCache = []

    def getClosestTemporary(self, config):
        minDist, closest = float('inf'), None
        for tconfig in self._temporalMiniCache:
            tdist = self._cspaceSampler.distance(config, tconfig)
            if tdist < minDist:
                minDist = tdist
                closest = tconfig
        return minDist, closest

    def addApproximate(self, config):
        cid = len(self._approximateConfigs)
        self._approximateConfigs.append(config)
        pointList = self._makeCoordinates(config)
        self._approximateIndex.insert(cid, pointList)

    def drawRandomApproximate(self):
        idx = len(self._approximateConfigs) - 1
        if idx == -1:
            return None
        config = self._approximateConfigs.pop()
        self._approximateIndex.delete(idx, self._makeCoordinates(config))
        return config

    def _makeCoordinates(self, config):
        pointList = list(config)
        pointList = map(lambda x, y: math.sqrt(x) * y, self._scalingFactors, pointList)
        pointList += pointList
        return pointList

class FreeSpaceProximityHierarchyNode(object):
    def __init__(self, goalNode, config=None, initialTemp=0.0, activeChildrenCapacity=20):
        self._goalNodes = []
        self._goalNodes.append(goalNode)
        self._activeGoalNodeIdx = 0
        self._children = []
        self._childrenContactLabels = []
        self._activeChildren = []
        self._inactiveChildren = []
        self._t = initialTemp
        self._T = 0.0
        self._T_c = 0.0
        self._T_p = 0.0
        self._numLeavesInBranch = 1 if goalNode.is_leaf() else 0
        self._numTimesSampled = 0
        self._activeChildrenCapacity = activeChildrenCapacity
        self._configs = []
        self._configs.append(None)
        self._configsRegistered = []
        self._configsRegistered.append(True)
        self._parent = None
        # INVARIANT: _configs[0] is always None
        #            _goalNodes[0] is hierarchy node that has all information
        #            _configsRegistered[i] is False iff _configs[i] is valid and new
        #            _goalNodes[i] and _configs[i] belong together for i > 0
        if config is not None:
            self._goalNodes.append(goalNode)
            self._configs.append(config)
            self._activeGoalNodeIdx = 1
            self._configsRegistered.append(not goalNode.is_valid())

    def getT(self):
        return self._T

    def gett(self):
        return self._t

    def getTc(self):
        return self._T_c

    def getTp(self):
        return self._T_p

    def setT(self, value):
        self._T = value
        assert self._T > 0.0

    def sett(self, value):
        self._t = value
        assert self._t > 0.0 or self.isLeaf() and not self.hasConfiguration()

    def setTp(self, value):
        self._T_p = value

    def setTc(self, value):
        self._T_c = value
        assert self._T_c > 0.0

    def updateActiveChildren(self, upTemperatureFunc):
        # For completeness, reactivate a random inactive child:
        if len(self._inactiveChildren) > 0:
            reactivatedChild = self._inactiveChildren.pop()
            upTemperatureFunc(reactivatedChild)
            self._activeChildren.append(reactivatedChild)

        while len(self._activeChildren) > self._activeChildrenCapacity:
            p = random.random()
            sumTemp = 0.0
            for child in self._activeChildren:
                sumTemp += 1.0 / child.getTc()

            assert sumTemp > 0.0
            acc = 0.0
            i = 0
            while acc < p:
                acc += 1.0 / self._activeChildren[i].getTc() * 1.0 / sumTemp
                i += 1
            deletedChild = self._activeChildren[max(i-1, 0)]
            self._activeChildren.remove(deletedChild)
            self._inactiveChildren.append(deletedChild)
            logging.debug('[FreeSpaceProximityHierarchyNode::updateActiveChildren] Removing child with ' + \
                          'temperature ' +  str(deletedChild.getT()) + '. It had index ' + str(i))
        assert len(self._children) == len(self._inactiveChildren) + len(self._activeChildren)

    def addChild(self, child):
        self._children.append(child)
        self._activeChildren.append(child)
        self._childrenContactLabels.append(child.getContactLabels())
        child._parent = self
        if child.isLeaf():
            self._numLeavesInBranch += 1
            parent = self._parent
            while parent is not None:
                parent._numLeavesInBranch += 1
                parent = parent._parent

    def getNumLeavesInBranch(self):
        return self._numLeavesInBranch

    def getMaxNumLeavesInBranch(self):
        return self._goalNodes[0].get_num_possible_leaves()

    def getMaxNumChildren(self):
        return self._goalNodes[0].get_num_possible_children()

    def getCoverage(self):
        if self.isLeaf():
            return 1.0
        return self.getNumChildren() / float(self.getMaxNumChildren())

    def getParent(self):
        return self._parent

    def getQuality(self):
        return self._goalNodes[0].get_quality()

    def hasChildren(self):
        return self.getNumChildren() > 0

    def getNumChildren(self):
        return len(self._children)

    def getChildren(self):
        return self._children

    def getContactLabels(self):
        return self._goalNodes[0].get_labels()

    def getChildrenContactLabels(self):
        return self._childrenContactLabels

    def getActiveChildren(self):
        return self._activeChildren

    def getUniqueLabel(self):
        return self._goalNodes[0].get_unique_label()

    def getConfigurations(self):
        """ Returns all configurations stored for this hierarchy node."""
        return self._configs[1:]

    def getValidConfigurations(self):
        """ Returns only valid configurations """
        validConfigs = []
        for idx in range(1, len(self._goalNodes)):
            if self._goalNodes[idx].is_valid():
                validConfigs.append(self._configs[idx])
        return validConfigs

    def get_depth(self):
        return self._goalNodes[0].get_depth()

    def isRoot(self):
        return self._goalNodes[0].get_depth() == 0

    def setActiveConfiguration(self, idx):
        assert idx in range(len(self._goalNodes)-1)
        self._activeGoalNodeIdx = idx + 1

    def getActiveConfiguration(self):
        if self._activeGoalNodeIdx in range(len(self._configs)):
            return self._configs[self._activeGoalNodeIdx]
        else:
            return None

    def hasConfiguration(self):
        return len(self._configs) > 1

    def getNewValidConfigs(self):
        unregisteredGoals = []
        unregisteredApprox = []
        for i in range(1, len(self._configs)):
            if not self._configsRegistered[i]:
                if self._goalNodes[i].is_goal():
                    unregisteredGoals.append((self._configs[i], self._goalNodes[i].get_hand_config()))
                else:
                    unregisteredApprox.append(self._configs[i])
                self._configsRegistered[i] = True
        return unregisteredGoals, unregisteredApprox

    def isGoal(self):
        return self._goalNodes[self._activeGoalNodeIdx].is_goal() and self.isValid()

    def isValid(self):
        bIsValid = self._goalNodes[self._activeGoalNodeIdx].is_valid()
        if not bIsValid:
            assert not reduce(lambda x, y: x or y, [x.is_valid() for x in self._goalNodes], False)
        return self._goalNodes[self._activeGoalNodeIdx].is_valid()

    def isExtendible(self):
        return self._goalNodes[0].is_extendible()

    def isLeaf(self):
        return not self.isExtendible()

    def isAllCovered(self):
        if self.isLeaf():
            return True
        return self.getNumChildren() == self.getMaxNumChildren()

    def getGoalSamplerHierarchyNode(self):
        return self._goalNodes[self._activeGoalNodeIdx]

    def toSampleData(self, idNum=-1):
        return SampleData(self._configs[self._activeGoalNodeIdx],
                          data=self._goalNodes[self._activeGoalNodeIdx].get_hand_config(),
                          idNum=idNum)

    def addGoalSample(self, sample):
        sampleConfig = sample.getConfiguration()
        if sampleConfig is None:
            return
        bConfigKnown = False
        for config in self._configs[1:]:
            bConfigKnown = numpy.linalg.norm(config - sampleConfig) < NUMERICAL_EPSILON
            if bConfigKnown:
                return
        self._configs.append(sample.getConfiguration())
        self._goalNodes.append(sample.hierarchyInfo)
        self._configsRegistered.append(not sample.isValid())

    def sampled(self):
        self._numTimesSampled += 1

    def getNumTimesSampled(self):
        return self._numTimesSampled


class FreeSpaceProximitySampler(object):
    def __init__(self, goalSampler, cfreeSampler, k=4, numIterations=10,
                 minNumIterations=8,
                 returnApproximates=True,
                 connectedWeight=10.0, freeSpaceWeight=5.0, debugDrawer=None):
        self._goalHierarchy = goalSampler
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
        self._numIterations = max(1, goalSampler.get_max_depth()) * [numIterations]
        self._minNumIterations = minNumIterations
        self._connectedWeight = connectedWeight
        self._freeSpaceWeight = freeSpaceWeight
        self._connectedSpace = None
        self._nonConnectedSpace = None
        self._debugDrawer = debugDrawer
        self._cfreeSampler = cfreeSampler
        self._labelCache = {}
        self._goalLabels = []
        self._rootNode = FreeSpaceProximityHierarchyNode(goalNode=self._goalHierarchy.get_root(),
                                                         initialTemp=self._freeSpaceWeight)
        maxDist = numpy.linalg.norm(cfreeSampler.getUpperBounds() - cfreeSampler.getLowerBounds())
        self._minConnectionChance = self._distanceKernel(maxDist)
        self._minFreeSpaceChance = self._distanceKernel(maxDist)
        self._bReturnApproximates = returnApproximates

    def clear(self):
        logging.debug('[FreeSpaceProximitySampler::clear] Clearing caches etc')
        self._connectedSpace = None
        self._nonConnectedSpace = None
        self._labelCache = {}
        self._goalLabels = []
        self._rootNode = FreeSpaceProximityHierarchyNode(goalNode=self._goalHierarchy.get_root(),
                                                         initialTemp=self._freeSpaceWeight)
        if self._debugDrawer is not None:
            self._debugDrawer.clear()

    def object_reloaded(self):
        self.clear()
        self._numIterations = self._goalHierarchy.get_max_depth() * [self._numIterations[0]]


    def getNumGoalNodesSampled(self):
        return len(self._labelCache)

    def getQuality(self, sampleData):
        idx = sampleData.getId()
        node = self._labelCache[self._goalLabels[idx]]
        return node.getQuality()

    def setConnectedSpace(self, connectedSpace):
        self._connectedSpace = connectedSpace

    def setNonConnectedSpace(self, nonConnectedSpace):
        self._nonConnectedSpace = nonConnectedSpace

    def _getHierarchyNode(self, goalSample):
        label = goalSample.hierarchyInfo.get_unique_label()
        bNew = False
        hierarchyNode = None
        if label in self._labelCache:
            hierarchyNode = self._labelCache[label]
            hierarchyNode.addGoalSample(goalSample)
            logging.debug('[FreeSpaceProximitySampler::_getHierarchyNode] Sampled a cached node!')
        else:
            hierarchyNode = FreeSpaceProximityHierarchyNode(goalNode=goalSample.hierarchyInfo,
                                                            config=goalSample.getConfiguration())
            self._labelCache[label] = hierarchyNode
            bNew = True
        return (hierarchyNode, bNew)

    def _filterRedundantChildren(self, children):
        labeledChildren = []
        filteredChildren = []
        for child in children:
            labeledChildren.append((child.getUniqueLabel(), child))
        labeledChildren.sort(key=lambda x: x[0])
        prevLabel = ''
        for labeledChild in labeledChildren:
            if labeledChild[0] == prevLabel:
                continue
            filteredChildren.append(labeledChild[1])
            prevLabel = labeledChild[0]
        return filteredChildren

    def _sampleKChildren(self, node, depth):
        children = []
        goalNode = node.getGoalSamplerHierarchyNode()
        numValids = 0
        for c in range(self._k):
            self._goalHierarchy.set_max_iter(self._numIterations[depth])
            goalSample = self._goalHierarchy.sample_warm_start(hierarchy_node=goalNode, depth_limit=1)
            if goalSample.hierarchyInfo.is_goal() and goalSample.hierarchyInfo.is_valid():
                logging.debug('[FreeSpaceProximitySampler::_sampleKChildren] We sampled a goal here!!!')
            if goalSample.hierarchyInfo.is_valid():
                numValids += 1
                logging.debug('[FreeSpaceProximitySampler::_sampleKChildren] Valid sample here!')
            (hierarchyNode, bNew) = self._getHierarchyNode(goalSample)
            children.append(hierarchyNode)
            if bNew:
                node.addChild(hierarchyNode)
            # raw_input()
        children = self._filterRedundantChildren(children)
        if len(children) < self._k:
            logging.warn('[FreeSpaceProximitySampler::_sampleKChildren] Failed at sampling k=%i children. Sampled only %i children' % (self._k, len(children)))
        logging.debug('[FreeSpaceProximitySampler::_sampleKChildren] We sampled %i valid children.'
                      %numValids)
        return children

    def _computeConnectionChance(self, config):
        (dist, nearestConfig) = self._connectedSpace.getNearestConfiguration(config)
        if nearestConfig is None:
            return self._minConnectionChance
        return self._distanceKernel(dist)

    def _computeFreeSpaceChance(self, config):
        (dist, nearestConfig) = self._nonConnectedSpace.getNearestConfiguration(config)
        if nearestConfig is None:
            return self._minFreeSpaceChance
        return self._distanceKernel(dist)

    def _distanceKernel(self, dist):
        return math.exp(-dist)

    def _updateTemperatures(self, node):
        logging.debug('[FreeSpaceProximitySampler::_updateTemperatures] Updating temperatures')
        self._T(node)

    def _t(self, node):
        if node.isRoot():
            node.sett(self._freeSpaceWeight)
            return self._freeSpaceWeight
        if not node.hasConfiguration() and node.isExtendible():
            parent = node.getParent()
            assert parent is not None
            tN = parent.getCoverage() * parent.gett()
            node.sett(tN)
            return tN
        elif not node.hasConfiguration() and node.isLeaf():
            # TODO: we should actually set this to 0.0 and prune covered useless branches
            minimalTemp = self._minConnectionChance + self._minFreeSpaceChance
            node.sett(minimalTemp)
            return minimalTemp
        maxTemp = 0.0
        configId = 0
        for config in node.getConfigurations():
            temp = self._connectedWeight * self._computeConnectionChance(config) \
                   + self._freeSpaceWeight * self._computeFreeSpaceChance(config)
            if maxTemp < temp:
                node.setActiveConfiguration(configId)
                maxTemp = temp
            configId += 1
        node.sett(maxTemp)
        assert (node.isValid() and maxTemp >= self._freeSpaceWeight) or not node.isValid()
        return node.gett()

    def _T(self, node):
        tempsChildren = 0.0
        tNode = self._t(node)
        avgChildTemp = tNode
        if len(node.getActiveChildren()) > 0:
            for child in node.getActiveChildren():
                tempsChildren += self._T(child)
            avgChildTemp = tempsChildren / float(len(node.getActiveChildren()))
        node.setT((tNode + avgChildTemp) / 2.0)
        self._T_c(node)
        self._T_p(node)
        return node.getT()

    def _T_c(self, node):
        mod_branch_coverage = node.getNumLeavesInBranch() / (node.getMaxNumLeavesInBranch() + 1)
        T_c = node.getT() * (1.0 - mod_branch_coverage)
        node.setTc(T_c)
        return T_c

    def _T_p(self, node):
        T_p = node.getT() * (1.0 - node.getCoverage())
        node.setTp(T_p)
        return T_p

    def _pickRandomNode(self, p, nodes):
        modifiedTemps = [self._T_c(x) for x in nodes]
        accTemp = sum(modifiedTemps)
        assert accTemp > 0.0
        i = 0
        acc = 0.0
        while p > acc:
            acc += modifiedTemps[i] / accTemp
            i += 1

        idx = max(i-1, 0)
        otherNodes = nodes[:idx]
        if idx + 1 < len(nodes):
            otherNodes.extend(nodes[idx+1:])
        return (nodes[idx], otherNodes)

    def _updateApproximate(self, children):
        for child in children:
            goalConfigs, approxConfigs = child.getNewValidConfigs()
            assert len(goalConfigs) == 0
            for config in approxConfigs:
                self._nonConnectedSpace.addApproximate(config)

    def _pickRandomApproximate(self):
        randomApproximate = self._nonConnectedSpace.drawRandomApproximate()
        logging.debug('[FreeSpaceProximitySampler::_pickRandomApproximate] ' + str(randomApproximate))
        return randomApproximate

    def _addTemporary(self, children):
        for node in children:
            if node.isValid():
                self._nonConnectedSpace.addTemporary(node.getValidConfigurations())

    def _clearTemporary(self):
        self._nonConnectedSpace.clearTemporaryCache()

    def _pickRandomChild(self, node):
        if not node.hasChildren():
            return None

        node.updateActiveChildren(self._updateTemperatures)
        p = random.random()
        (child, otherChildren) = self._pickRandomNode(p, node.getActiveChildren())
        return child

    def _shouldDescend(self, parent, child):
        if child is None:
            return False
        if not child.isExtendible():
            return False
        if parent.isAllCovered():
            return True
        p = random.random()
        tP = self._T_p(parent)
        sumTemp = tP + self._T_c(child)
        if p <= tP / sumTemp:
            return False
        return True

    def _sampleChild(self, node):
        goalNode = node.getGoalSamplerHierarchyNode()
        depth = node.get_depth()
        numIterations = int(self._minNumIterations + \
                        node.getT() / (self._connectedWeight + self._freeSpaceWeight) * \
                        (self._numIterations[depth] - self._minNumIterations))
        # numIterations = max(self._minNumIterations, int(numIterations))
        assert numIterations >= self._minNumIterations
        assert numIterations <= self._numIterations[depth]
        self._goalHierarchy.set_max_iter(numIterations)
        doPostOpt = depth == self._goalHierarchy.get_max_depth() - 1
        childrenContactLabes = node.getChildrenContactLabels()
        goalSample = self._goalHierarchy.sample_warm_start(hierarchy_node=goalNode, depth_limit=1,
                                                           label_cache=childrenContactLabes,
                                                           post_opt=doPostOpt)
        if goalSample.hierarchyInfo.is_goal() and goalSample.hierarchyInfo.is_valid():
            logging.debug('[FreeSpaceProximitySampler::_sampleChild] We sampled a goal here!!!')
        if goalSample.hierarchyInfo.is_valid():
            logging.debug('[FreeSpaceProximitySampler::_sampleChild] Valid sample here!')
        (hierarchyNode, bNew) = self._getHierarchyNode(goalSample)
        if bNew:
            node.addChild(hierarchyNode)
        node.sampled()
        return hierarchyNode

    def is_goal(self, sample):
        if sample.getId() >= 0:
            return True
        return False

    # def sample(self):
    #     currentNode = self._rootNode
    #     logging.debug('[FreeSpaceProximitySampler::sample] Starting to sample a new goal candidate' + \
    #                   ' - the conservative-lazy way')
    #     numSamplings = self._goalHierarchy.getMaxDepth()
    #     while numSamplings > 0:
    #         if self._debugDrawer is not None:
    #             self._debugDrawer.drawHierarchy(self._rootNode)
    #         logging.debug('[FreeSpaceProximitySampler::sample] Sampling random new child')
    #         newChild = self._sampleChild(currentNode)
    #         goalConfigs, approxConfigs = newChild.getNewValidConfigs()
    #         assert len(goalConfigs) + len(approxConfigs) <= 1
    #         if len(goalConfigs) > 0:
    #             self._goalIds += 1
    #             return SampleData(config=goalConfigs[0][0], data=goalConfigs[0][1], idNum=self._goalIds)
    #         elif len(approxConfigs) > 0:
    #             self._nonConnectedSpace.addApproximate(approxConfigs[0])
    #         # we always descend
    #         currentNode = self._pickRandomChild(currentNode, bUpdateTemperatures=True)
    #         numSamplings -= 1
    #
    #     if self._debugDrawer is not None:
    #         self._debugDrawer.drawHierarchy(self._rootNode)
    #     logging.debug('[FreeSpaceProximitySampler::sample] The search led to a dead end. Maybe there is' \
    #                   + 'sth in our approximate cache!')
    #     return SampleData(self._pickRandomApproximate())
        # return SampleData(None)

    def sample(self):
        currentNode = self._rootNode
        logging.debug('[FreeSpaceProximitySampler::sample] Starting to sample a new goal candidate' + \
                      ' - the lazy way')
        numSamplings = self._k
        temperaturesInvalid = True
        while numSamplings > 0:
            if self._debugDrawer is not None:
                self._debugDrawer.drawHierarchy(self._rootNode)
            logging.debug('[FreeSpaceProximitySampler::sample] Picking random cached child')
            if temperaturesInvalid:
                self._updateTemperatures(currentNode)
            child = self._pickRandomChild(currentNode)
            if self._shouldDescend(currentNode, child):
                currentNode = child
                temperaturesInvalid = False
            elif currentNode.isAllCovered():
                # There are no children left to sample, sample the null space of the child instead
                # TODO: It depends on our IK solver on what we have to do here. If the IK solver is complete,
                # we do not resample non-goal nodes. If it is not complete, we would need to give them
                # another chance here. Hence, we would also need to set the temperatures of such nodes
                # to sth non-zero
                if child.isGoal():
                    logging.warn('[FreeSpaceProximitySampler::sample] Pretending to sample null space')
                    # TODO actually sample null space here and return new configuration or approx
                    temperaturesInvalid = True
                numSamplings -= 1
            else:
                newChild = self._sampleChild(currentNode)
                temperaturesInvalid = True
                numSamplings -= 1
                goalConfigs, approxConfigs = newChild.getNewValidConfigs()
                assert len(goalConfigs) + len(approxConfigs) <= 1
                # if newChild.isValid() and newChild.isGoal():
                if len(goalConfigs) > 0:
                    self._goalLabels.append(newChild.getUniqueLabel())
                    return SampleData(config=goalConfigs[0][0], data=goalConfigs[0][1],
                                      idNum=len(self._goalLabels)-1)
                # elif newChild.isValid():
                elif len(approxConfigs) > 0:
                    self._nonConnectedSpace.addApproximate(approxConfigs[0])
                # self._computeTemperatures(currentNode)

        if self._debugDrawer is not None:
            self._debugDrawer.drawHierarchy(self._rootNode)
        logging.debug('[FreeSpaceProximitySampler::sample] The search led to a dead end. Maybe there is ' \
                      + 'sth in our approximate cache!')
        if self._bReturnApproximates:
            return SampleData(self._pickRandomApproximate())
        return SampleData(None)

    # def sample(self):
    #     nextNode = self._rootNode
    #     logging.debug('[FreeSpaceProximitySampler::sample] Starting to sample a new goal candidate')
    #     for d in range(0, self._goalHierarchy.getMaxDepth()):
    #         if self._debugDrawer is not None:
    #             self._debugDrawer.drawHierarchy(self._rootNode)
    #         logging.debug('[FreeSpaceProximitySampler::sample] Sampling k children on level %i' %d)
    #         children = self._sampleKChildren(nextNode, d)
    #         self._addTemporary(children)
    #         for child in children:
    #             self._computeT(child)
    #         p = random.random()
    #         logging.debug('[FreeSpaceProximitySampler::sample] Computed temperatures, picking random child')
    #         (nextNode, otherChildren) = self._pickRandomNode(p, children)
    #         if nextNode.isGoal():
    #             logging.debug('[FreeSpaceProximitySampler::sample] The child we picked is a goal!')
    #             self._goalIds += 1
    #             self._updateApproximate(otherChildren)
    #             return nextNode.toSampleData(idNum=self._goalIds)
    #         elif d < self._goalHierarchy.getMaxDepth() - 1:
    #             logging.debug('[FreeSpaceProximitySampler::sample] Going further down, updating approx!')
    #             self._updateApproximate(children)
    #         else:
    #             logging.debug('[FreeSpaceProximitySampler::sample] Reached the bottom of hierarchy ' \
    #                           + 'approx!')
    #             self._updateApproximate(otherChildren)
    #         self._clearTemporary()
    #     if self._debugDrawer is not None:
    #         self._debugDrawer.drawHierarchy(self._rootNode)
    #     if nextNode.isValid():
    #         logging.debug('[FreeSpaceProximitySampler::sample] We found an approximate goal!')
    #         return nextNode.toSampleData()
    #     logging.debug('[FreeSpaceProximitySampler::sample] The search led to a dead end. Maybe there is' \
    #                   + 'sth in our approximate cache!')
    #     return SampleData(self._pickRandomApproximate())

    def debugDraw(self):
        # nodesToUpdate = []
        # nodesToUpdate.extend(self._rootNode.getChildren())
        # while len(nodesToUpdate) > 0:
            # nextNode = nodesToUpdate.pop()
            # nodesToUpdate.extend(nextNode.getChildren())
        if self._debugDrawer is not None:
            self._updateTemperatures(self._rootNode)
            self._debugDrawer.drawHierarchy(self._rootNode)
