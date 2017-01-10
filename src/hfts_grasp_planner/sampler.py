#!/usr/bin/env python
""" This module contains a general hierarchically organized goal region sampler. """

import ast
import logging
import math
import numpy
import random
from blist import sortedlist

from rtree import index

from src.hfts_grasp_planner.rrt import SampleData

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
            return self.hierarchyInfo.isGoal()
        return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{SamplingResult:[Config=" + str(self.configuration) + "; Info=" + str(self.hierarchyInfo) + "]}"


class HotRegion:
    def __init__(self, samplingResult):
        self.samplingResult = samplingResult
        self.temperature = 6.0  # TODO: could be based on the branching factor in the hierarchy


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

class GoalSampler:
    """ Goal sampler for hierarchically approximatable goal regions """
    def __init__(self, goalRegionApproximation, pHotRegions=0.8, pCachedSample=0.2):
        # The hierarchical goal region approximation that is expensive to query
        self.goalRegionApprox = goalRegionApproximation
        # A cache of previously sampled approximate goal configurations (stores SamplingResults)
        # that weren't refined.
        self.sampleCache = sortedlist([])
        # A cache that stores all sampling results.
        self.allSamples = []
        # A cache of hot regions (stores SamplingResults that have been reached by the tree)
        self.hotRegions = sortedlist([])
        # Probability of sampling from the hot regions, i.e. to perform a warm start
        self.pHotRegions = pHotRegions
        # Probability of re-using a previously sampled yet unreached approximate goal
        self.pCachedSample = pCachedSample
        # Accumulated temperature
        self.accTemperature = 0.0

    def clear(self):
        self.sampleCache = sortedlist([])
        self.allSamples = []
        self.hotRegions = sortedlist([])
        self.accTemperature = 0.0

    def _adjustTemperature(self, diffT):
        self.accTemperature = max(0.0, self.accTemperature + diffT)

    def _sampleHotRegions(self):
        # Each hot region has a temperature that decreases everytime we sample it
        p = random.random()
        summedWeight = 0.0
        index = 0
        while summedWeight < p:
            summedWeight = summedWeight + self.hotRegions[index].temperature / self.accTemperature
            index = index + 1

        return self.hotRegions[index - 1]

    def _updateHotRegions(self, hotRegion):
        if not hotRegion.temperature > 0.0:
            self.hotRegions.remove(hotRegion)

    def _sampleHotRegion(self, hotRegion, seedIk=None):
        sampleR = self.goalRegionApprox.sampleWarmStart(hotRegion.samplingResult.hierarchyInfo, 1,
                                                        seedIk=seedIk)
        sampleR.bOriginatesFromHotRegion = True
        hotRegion.temperature = hotRegion.temperature - 1.0
        self._adjustTemperature(-1.0)
        return sampleR

    def sample(self, bReuseCachedSamples=True, bHotRegionsOnly=False):
        """ Samples the approximate goal region """
        cacheResult = True

        if bHotRegionsOnly and len(self.hotRegions) == 0:
            return None

        # First, let's roll a dice what we wanna do
        p = random.random()
        if (p < self.pHotRegions or bHotRegionsOnly) and len(self.hotRegions) > 0:
            # Let's sample from a hot region. That is a region where we know we connected the RRT
            # already to. We can sample from this region again by performing a warm start
            # Choose a random hot region
            hotRegion = self._sampleHotRegions()
            # now do the warm start and we get a new sample which is created from a deeper hierarchy level
            sampleResult = self._sampleHotRegion(hotRegion)
            # update the hot regions list, e.g. remove hot regions that are cooled down
            self._updateHotRegions(hotRegion)
        elif p < self.pCachedSample + self.pHotRegions and len(self.sampleCache) > 0 and bReuseCachedSamples:
            # re-use a previously sampled goal sample that was not reachable before
            # i.e. it might be reachable by now and we get this for free
            sampleId = random.choice(self.sampleCache)
            sampleResult = self.allSamples[sampleId]
            cacheResult = False  # in this case we don't need to cache the result
        else:
            # Else, just get another sample from the roughest approximation layer
            sampleResult = self.goalRegionApprox.sample(1)

        if not sampleResult.isValid():
            # raise ValueError('Could not sample goal region. Is it empty?')
            cacheResult = False
            print 'WARNING: Goal region sample was None'

        if cacheResult:
            # Let's cache the query to save sampling and to remember it
            sampleResult.cacheId = len(self.allSamples)
            self.allSamples.append(sampleResult)
            self.sampleCache.add(sampleResult.cacheId)
        return sampleResult.toSampleData()

    def isApproxGoal(self, sample):
        """ Checks whether the given sample is an approximate goal.
            While checking this property might be cheap for some applications, it is definitely
            not cheap for fingertip grasp planning. Therefore, by default, we just check
            whether we sampled sample before. As a consequence, a sample that might actually
            be a goal, but it wasn't sampled by this sampler, is falsely classified as non-goal.
        """
        return self._isCachedSample(sample)
        # alternatively we query the goal region approximation here
        # TODO: can we decide here somehow, how deep we can go?
        # self.goalRegionApprox.containsApprox(qReachable, 1)

    def isGoal(self, sample):
        """ Checks whether sample is a real goal. """
        if not self._isCachedSample(sample):
            return False
        key = sample.getId()
        return self.goalRegionApprox.isGoal(self.allSamples[key])

    def _isCachedSample(self, sample):
        key = sample.getId()
        return key is not None and key >= 0 and key < len(self.allSamples)

    def refineGoalSampling(self, sample):
        """ Tells the sampler that the sample is now in the tree and we wish to explore that area in more detail """
        if not self._isCachedSample(sample):
            raise ValueError('The given sample has not been sampled before.')

        key = sample.getId()
        # The sample has been reached by the planner, no need to cache it for sampling anymore.
        self.sampleCache.remove(key)

        # add hot region
        newHotRegion = HotRegion(self.allSamples[key])
        self.hotRegions.add(newHotRegion)
        self._adjustTemperature(newHotRegion.temperature)
        # print len(self.hotRegions)

    def refineAndSample(self, sample):
        """ First refines the hot region where the sample originates from and then samples the new
            hot region. Returns the new sample."""
        if not self._isCachedSample(sample):
            raise ValueError('The given sample has not been sampled before.')

        key = sample.getId()
        # The sample has been reached by the planner, no need to cache it for sampling anymore.
        self.sampleCache.remove(key)

        # add hot region
        newHotRegion = HotRegion(self.allSamples[key])
        self.hotRegions.add(newHotRegion)
        self._adjustTemperature(newHotRegion.temperature)
        # Sample the new hot region
        sampleResult = self._sampleHotRegion(newHotRegion, seedIk=sample.getConfiguration())
        self._updateHotRegions(newHotRegion)
        if sampleResult.isValid():
            sampleResult.cacheId = len(self.allSamples)
            self.allSamples.append(sampleResult)
            self.sampleCache.add(sampleResult.cacheId)
        return sampleResult.toSampleData()

    def goalIsFromHotRegion(self, sampleData):
        if not self._isCachedSample(sampleData):
            raise ValueError('The given sample has not been sampled before.')
        key = sampleData.getId()
        return self.allSamples[key].bOriginatesFromHotRegion

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
        if self.hierarchyInfo.isGoal() and self.hierarchyInfo.isValid():
            return 1.5
        if self.hierarchyInfo.isValid():
            return 1.0
        return 0.0

    def isGoal(self):
        return self.hierarchyInfo.isGoal()

    def getActiveConfiguration(self):
        return self.config

    def addChild(self, child):
        self.children.append(child)

class NaiveGoalSampler:
    def __init__(self, goalRegion, numIterations=40, debugDrawer=None):
        self.goalRegion = goalRegion
        self.depthLimit = goalRegion.getMaxDepth()
        self.goalRegion.setMaxIter(numIterations)
        self._debugDrawer = debugDrawer
        self.clear()

    def clear(self):
        self.cache = []
        self._rootNode = SimpleHierarchyNode(None, self.goalRegion.getRoot())
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
        if not mySample.isValid() or not mySample.isGoal():
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
            return self.goalRegion.isGoal(self.cache[sample.getId()])
        return False
            # return True
        # return self.goalRegion.isGoalConfiguration(qReachable)

# Minimal temperature that a node can have
MINIMAL_TEMPERATURE = 1.0
# Low temperature for nodes that are infeasible(invalid)
LOW_TEMPERATURE = 1.0
# Medium temperature for nodes that are valid
MEDIUM_TEMPERATURE = 5.0
# High temperature for nodes that were connected to the RRT
HIGH_TEMPERATURE = 15.0
# Cool down value when a node is sampled
SAMPLING_COOL_DOWN = 1.0


class HierarchyNode:
    def __init__(self, parent, plannerNode, initialTemp=MEDIUM_TEMPERATURE, sampleId=-1):
        self._parent = parent
        self._ownTemperature = initialTemp
        self._accChildrenTemperature = 0.0
        self._plannerNode = plannerNode
        self._sampleId = sampleId
        self._children = []

    def addChild(self, child):
        self._children.append(child)
        self._accChildrenTemperature += child.getTemperature()

    # def addNeighbor(self, neighbor):
    #     self._neighbors.append(neighbor)

    def getTemperature(self, includeChildren=True):
        if len(self._children) > 0 and includeChildren:
            childrenTemp = self._accChildrenTemperature / len(self._children)
        else:
            childrenTemp = 0.0
        return max(MINIMAL_TEMPERATURE, self._ownTemperature + childrenTemp)

    def getAccChildrenTemperature(self):
        return self._accChildrenTemperature

    def addTemperature(self, diff):
        self._ownTemperature = max(0.0, self._ownTemperature + diff)

    def updateChildTemperature(self, oldTemperature, newTemperature):
        self._accChildrenTemperature = self._accChildrenTemperature - oldTemperature + newTemperature

    def hasChildren(self):
        return len(self._children) > 0

    def getPlannerNode(self):
        return self._plannerNode

    def getChild(self, id):
        return self._children[id]

    def getSampleId(self):
        return self._sampleId

    def getNumChildren(self):
        return len(self._children)

    def getParent(self):
        return self._parent

    def getChildren(self):
        return self._children

    def hasExtendibleChildren(self):
        if self.hasChildren():
            return self._children[0].isExtendible()
        return False

    def isExtendible(self):
        return self._plannerNode.isExtendible()

    def setTemperature(self, temp):
        self._temperature = temp


class HierarchyGoalSampler:
    """ A goal sampler that samples goals from a tree-like hierarchy."""
    def __init__(self, goalHierarchy, cspaceSampler, debugDrawer=None):
        self._goalHierarchy = goalHierarchy
        self._rootNode = HierarchyNode(parent=None, plannerNode=self._goalHierarchy.getRoot())
        self._debugDrawer = debugDrawer
        self._allSamples = []
        self._hierarchyNodesMap = {}
        self._cspaceSampler = cspaceSampler

    def clear(self):
        self._rootNode = HierarchyNode(parent=None, plannerNode=self._goalHierarchy.getRoot())
        self._allSamples = []
        self._hierarchyNodesMap = {}

    def isGoal(self, sample):
        key = sample.getId()
        if not self._isValidSample(sample):
            raise ValueError('Invalid sample id %i!' % sample.getId())
        return self._goalHierarchy.isGoal(self._allSamples[key][0])

    def isApproxGoal(self, sample):
        # TODO
        raise NotImplementedError("isApproxGoal is not implemented yet!")

    def sample(self, bDummy=False, currentNode=None):
        """ Samples a new goal configuration from the hierarchical goal planner """
        logging.debug('[HierarchyGoalSampler::sample] Sampling a new goal')
        if currentNode is None:
            currentNode = self._rootNode
        # Move down in the hierarchy
        while True:
            # in case the current node has no children, we have to sample from it
            if not currentNode.hasExtendibleChildren():
                logging.debug('[HierarchyGoalSampler::sample] currentNode has no children,' +
                              'hence sampling it')
                return self._sampleHierarchyNode(currentNode)
            # else pick a random child with bias on hot ones
            logging.debug('[HierarchyGoalSampler::sample] Current node has children, picking a child')
            p = random.random()
            chosenChild = self._randomlyPickChild(p, currentNode)
            # let's roll a dice to see if we actually wanna go deeper
            p = random.random()
            # make the decision based on how hot the selected child is
            relTemperature = chosenChild.getTemperature() / (currentNode.getTemperature(includeChildren=False) +
                                                             chosenChild.getTemperature())
            if p > relTemperature:
                logging.debug('[HierarchyGoalSampler::sample] Not going further down. Sampling this node')
                return self._sampleHierarchyNode(currentNode)
            else:
                logging.debug('[HierarchyGoalSampler::sample] Decided to go further down in hierachy.')
                currentNode = chosenChild

    def refineGoalSampling(self, sample):
        """ Notifies the sampler to increase the probability of sampling the goal planner from
            the node for which the given sample is a representative. """
        key = sample.getId()
        if not self._isValidSample(sample):
            raise ValueError('Invalid sample id %i!') % sample.getId()
        treeNode = self._allSamples[key][1]
        if not treeNode._plannerNode.isExtendible():
            logging.debug('[HierarchyGoalSampler::refineGoalSampling] We are refining a bottom level sample'\
                          + '. The sample is a goal of bad quality and therefore not counted as goal.' \
                          + ' Since we can not sample from it, we increase this sample s parent s temperature' \
                          +  'instead!')
            treeNode = treeNode.getParent()

        oldTemperature = treeNode.getTemperature()
        treeNode.addTemperature(HIGH_TEMPERATURE)
        self._propagateChangesUp(treeNode, oldTemperature)
        return treeNode

    def refineAndSample(self, sample):
        """ First refines the goal sampling and then samples the node immediately.
            Returns a sample. """
        treeNode = self.refineGoalSampling(sample)
        return self.sample(currentNode=treeNode)

    def _sampleHierarchyNode(self, hNode):
        sampleResult = self._goalHierarchy.sampleWarmStart(hierarchyNode=hNode.getPlannerNode(), depthLimit=1)
        uniqueLabel = sampleResult.hierarchyInfo.getUniqueLabel()
        oldTemperature = hNode.getTemperature()
        bNewSample = True
        if uniqueLabel in self._hierarchyNodesMap:
            logging.debug('[HierarchyGoalSampler::_sampleHierarchyNode] We received a node we received' \
                          + 'before from the goal sampler: ' + uniqueLabel)
            oldSampleResult = self._allSamples[self._hierarchyNodesMap[uniqueLabel].getSampleId()][0]
            if not sampleResult.isValid() or \
                not oldSampleResult.isValid() or \
                self._cspaceSampler.configsAreEqual(sampleResult.getConfiguration(),
                                                   oldSampleResult.getConfiguration()):

                sampleResult = SamplingResult(None)
                bNewSample = False
            else:
                logging.debug('[HierarchyGoalSampler::_sampleHierarchyNode] However, the arm + hand ' \
                              + ' configuration is different. We consider it a new sample.')
        if bNewSample:
            logging.debug('[HierarchyGoalSampler::_sampleHierarchyNode] We received a new node. Adding it ' +
                          'to our goal sample hierarchy')
            initialTemp = MEDIUM_TEMPERATURE
            if not sampleResult.isValid():
                initialTemp = LOW_TEMPERATURE
            sampleId = len(self._allSamples)
            newNode = HierarchyNode(parent=hNode, plannerNode=sampleResult.hierarchyInfo,
                                    initialTemp=initialTemp, sampleId=sampleId)
            hNode.addChild(newNode)
            self._hierarchyNodesMap[uniqueLabel] = newNode
            sampleResult.cacheId = sampleId
            self._allSamples.append((sampleResult, newNode))
        hNode.addTemperature(-SAMPLING_COOL_DOWN)
        self._propagateChangesUp(hNode, oldTemperature)
        if self._debugDrawer is not None:
            self._debugDrawer.drawHierarchy(self._rootNode)
        return sampleResult.toSampleData()

    def _randomlyPickChild(self, p, parent):
        acctemp = 0.0
        chosenChildId = 0
        while acctemp < p and chosenChildId < parent.getNumChildren():
            acctemp += parent.getChild(chosenChildId).getTemperature() / parent.getAccChildrenTemperature()
            chosenChildId += 1
        return parent.getChild(chosenChildId-1)

    def _propagateChangesUp(self, node, oldNodeTemperature):
        # go recursively upwards and update temperatures
        parent = node.getParent()
        while parent is not None:
            oldParentTemperature = parent.getTemperature()
            parent.updateChildTemperature(oldNodeTemperature, node.getTemperature())
            oldNodeTemperature = oldParentTemperature
            node = parent
            parent = parent.getParent()

    def _isValidSample(self, sample):
        return sample.getId() >= 0 and sample.getId() < len(self._allSamples)


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
        self._numLeavesInBranch = 1 if goalNode.isLeaf() else 0
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
            self._configsRegistered.append(not goalNode.isValid())

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
        return self._goalNodes[0].getPossibleNumLeaves()

    def getMaxNumChildren(self):
        return self._goalNodes[0].getPossibleNumChildren()

    def getCoverage(self):
        if self.isLeaf():
            return 1.0
        return self.getNumChildren() / float(self.getMaxNumChildren())

    def getParent(self):
        return self._parent

    def getQuality(self):
        return self._goalNodes[0].getQuality()

    def hasChildren(self):
        return self.getNumChildren() > 0

    def getNumChildren(self):
        return len(self._children)

    def getChildren(self):
        return self._children

    def getContactLabels(self):
        return self._goalNodes[0].getContactLabels()

    def getChildrenContactLabels(self):
        return self._childrenContactLabels

    def getActiveChildren(self):
        return self._activeChildren

    def getUniqueLabel(self):
        return self._goalNodes[0].getUniqueLabel()

    def getConfigurations(self):
        """ Returns all configurations stored for this hierarchy node."""
        return self._configs[1:]

    def getValidConfigurations(self):
        """ Returns only valid configurations """
        validConfigs = []
        for idx in range(1, len(self._goalNodes)):
            if self._goalNodes[idx].isValid():
                validConfigs.append(self._configs[idx])
        return validConfigs

    def getDepth(self):
        return self._goalNodes[0].getDepth()

    def isRoot(self):
        return self._goalNodes[0].getDepth() == 0

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
                if self._goalNodes[i].isGoal():
                    unregisteredGoals.append((self._configs[i], self._goalNodes[i].getHandConfig()))
                else:
                    unregisteredApprox.append(self._configs[i])
                self._configsRegistered[i] = True
        return unregisteredGoals, unregisteredApprox

    def isGoal(self):
        return self._goalNodes[self._activeGoalNodeIdx].isGoal() and self.isValid()

    def isValid(self):
        bIsValid = self._goalNodes[self._activeGoalNodeIdx].isValid()
        if not bIsValid:
            assert not reduce(lambda x, y: x or y, [x.isValid() for x in self._goalNodes], False)
        return self._goalNodes[self._activeGoalNodeIdx].isValid()

    def isExtendible(self):
        return self._goalNodes[0].isExtendible()

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
                          data=self._goalNodes[self._activeGoalNodeIdx].getHandConfig(),
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
    def __init__(self, goalSampler, cfreeSampler, k=4, numIterations=None,
                 minNumIterations=8,
                 returnApproximates=True,
                 connectedWeight=10.0, freeSpaceWeight=5.0, debugDrawer=None):
        self._goalHierarchy = goalSampler
        self._k = k
        if numIterations is None:
            numIterations = goalSampler.getMaxDepth() * [10]
        elif type(numIterations) == int:
            numIterations = goalSampler.getMaxDepth() * [numIterations]
        elif type(numIterations) == list:
            numIterations = numIterations
        else:
            raise ValueError('numIterations has invalid type %s. Supported are int, list and None' %
                             str(type(numIterations)))
        self._numIterations = numIterations
        self._minNumIterations = minNumIterations
        self._connectedWeight = connectedWeight
        self._freeSpaceWeight = freeSpaceWeight
        self._connectedSpace = None
        self._nonConnectedSpace = None
        self._debugDrawer = debugDrawer
        self._cfreeSampler = cfreeSampler
        self._labelCache = {}
        self._goalLabels = []
        self._rootNode = FreeSpaceProximityHierarchyNode(goalNode=self._goalHierarchy.getRoot(),
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
        self._rootNode = FreeSpaceProximityHierarchyNode(goalNode=self._goalHierarchy.getRoot(),
                                                         initialTemp=self._freeSpaceWeight)
        if self._debugDrawer is not None:
            self._debugDrawer.clear()

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
        label = goalSample.hierarchyInfo.getUniqueLabel()
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
            self._goalHierarchy.setMaxIter(self._numIterations[depth])
            goalSample = self._goalHierarchy.sampleWarmStart(hierarchyNode=goalNode, depthLimit=1)
            if goalSample.hierarchyInfo.isGoal() and goalSample.hierarchyInfo.isValid():
                logging.debug('[FreeSpaceProximitySampler::_sampleKChildren] We sampled a goal here!!!')
            if goalSample.hierarchyInfo.isValid():
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
        depth = node.getDepth()
        numIterations = int(self._minNumIterations + \
                        node.getT() / (self._connectedWeight + self._freeSpaceWeight) * \
                        (self._numIterations[depth] - self._minNumIterations))
        # numIterations = max(self._minNumIterations, int(numIterations))
        assert numIterations >= self._minNumIterations
        assert numIterations <= self._numIterations[depth]
        self._goalHierarchy.setMaxIter(numIterations)
        doPostOpt = depth == self._goalHierarchy.getMaxDepth() - 1
        childrenContactLabes = node.getChildrenContactLabels()
        goalSample = self._goalHierarchy.sampleWarmStart(hierarchyNode=goalNode, depthLimit=1,
                                                         labelCache=childrenContactLabes,
                                                         postOpt=doPostOpt)
        if goalSample.hierarchyInfo.isGoal() and goalSample.hierarchyInfo.isValid():
            logging.debug('[FreeSpaceProximitySampler::_sampleChild] We sampled a goal here!!!')
        if goalSample.hierarchyInfo.isValid():
            logging.debug('[FreeSpaceProximitySampler::_sampleChild] Valid sample here!')
        (hierarchyNode, bNew) = self._getHierarchyNode(goalSample)
        if bNew:
            node.addChild(hierarchyNode)
        node.sampled()
        return hierarchyNode

    def isGoal(self, sample):
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
        logging.debug('[FreeSpaceProximitySampler::sample] The search led to a dead end. Maybe there is' \
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
