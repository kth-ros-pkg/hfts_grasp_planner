#! /usr/bin/python

import numpy as np
from math import exp
import openravepy as orpy
import transformations
from scipy.optimize import fmin_cobyla
from RobotiqLoader import RobotiqHand
import sys, time, logging, copy
import itertools, random
from HFTSMotion.orRobot.handBase import InvalidTriangleException
from sets import Set
from utils import objectFileIO
import rospy


class graspSampler:

    def __init__(self, verbose=False, numHops=4, vis=False):

        self._verbose = verbose
        self._samplerViewer = vis
        self._orEnv = orpy.Environment() # create openrave environment
        self._orEnv.SetDebugLevel(orpy.DebugLevel.Fatal)
        self._orEnv.GetCollisionChecker().SetCollisionOptions(orpy.CollisionOptions.Contacts)
        if vis:
            self._orEnv.SetViewer('qtcoin') # attach viewer (optional)
        self._handLoaded = False

        self._mu = 2.
        self._alpha = 2.
        self._maxIters = 40
        self._hops = numHops
        self._ita = 0.001

        self._graspConf = None
        self._preGraspConf = None
        self._graspPos = None
        self._graspPose = None
        self._graspContacts = None
        self._armConf = None
        self._handPose_lab = None


    def __del__(self):
        orpy.RaveDestroy()
    
    
    def getOrHand(self):
        return self._robot

    def loadHand(self, handFile):
        if not self._handLoaded:
            self._robot = RobotiqHand(env = self._orEnv, handFile = handFile)
            self._handMani = self._robot.getHandMani()
            self._contactN = self._robot.getContactNumber()
            shift = transformations.identity_matrix()
            shift[0,-1] = 0.2
            self._robot.SetTransform(shift)
            rospy.loginfo('Hand loaded in OpenRAVE environment')
            self._handLoaded = True

    def loadObj(self, dataPath, objId):

        objectIO = objectFileIO(dataPath, objId)
        self._dataLabeled, self._levels = objectIO.getHFTS()
        self._nLevel = len(self._levels)
        self._objLoaded = self._orEnv.Load(dataPath + '/' + objId + '/objectModel' + objectIO.getObjFileExtension())
        self._obj = self._orEnv.GetKinBody('objectModel')
        self._objCOM = objectIO.getObjCOM()
        rospy.loginfo('Object loaded in OpenRAVE environment')
        self._objLoaded = True
# 
# 

    def initPlanner(self):
        if self._verbose:
            print 'initializing planner'
        topNodes = self._levels[0] + 1
        contactLabel = []

        for i in range(self._contactN):
            contactLabel.append([random.choice(range(topNodes))])

        return contactLabel




    def clearConfigCache(self):

        self._graspConf = None
        self._preGraspConf = None
        self._graspPos = None
        self._graspPose = None
        self._graspContacts = None
        shift = transformations.identity_matrix()
        shift[0,-1] = 0.2
        self._robot.SetTransform(shift)
        self.handles = []
        self.tipPNHandler = []

    def sampleGrasp(self, node, depthLimit, postOpt=False, openHandOffset=0.1):

        assert depthLimit >= 0
        if node.getDepth() >= self._nLevel:
            raise ValueError('graspSampler::sampleGrasp input node has an invalid depth')
            
            
        if node.getDepth() + depthLimit >= self._nLevel:
            depthLimit = self._nLevel - node.getDepth() - 1 # cap

        if node.getDepth() == 0: # at root
            contactLabel = self.initPlanner()
            bestO = -np.inf ## need to also consider non-root nodes
        else:
            # If we are not at a leaf node, go down in the hierarchy
            contactLabel = copy.deepcopy(node.getLabels())
            bestO, contactLabel = self.extendSolution(contactLabel)
            depthLimit -= 1

        self.clearConfigCache()
        

        while True:
            # just do it until depthLimit is reached
            for iter_now in range(self._maxIters):
                labels_tmp = self.getSiblingLabels(currLabels=contactLabel)
                s_tmp, r_tmp, o_tmp = self.evaluateGrasp(labels_tmp)

                if self.shcEvaluation(o_tmp, bestO):
                    contactLabel = labels_tmp
                    bestO = o_tmp
                    if self._verbose:
                        print '---------------------------------------------------'
                        print 'improved at level: %d, iter: %d' % (depthLimit, iter_now)
                        print s_tmp, r_tmp, o_tmp
                        print '---------------------------------------------------'
                    
            # extending to next level
            if depthLimit > 0:
                bestO, contactLabel = self.extendSolution(contactLabel)
                depthLimit -= 1
            else: # consider output
                self.composeGraspInfo(contactLabel)
                if self.executeInOR(postOpt = postOpt):
                    return HFTSNode(labels = contactLabel, handConf = self._robot.GetDOFValues(),
                                    handTransform = self._robot.GetTransform(), objTransform = self._obj.GetTransform(),
                                    isGoal = True)
                else:
                    return HFTSNode()
                    

    def plotClusters(self, contactLabels):

        if not self._samplerViewer:
            return
        self.cloudPlot = []
        colors = [np.array((1,0,0)), np.array((0,1,0)), np.array((0,0,1))]

        for i in range(3):
            label = contactLabels[i]

            level = len(label) - 1 # indexed from 0
            idx = np.where((self._dataLabeled[:, 6:7 + level] == label).all(axis=1))
            points = [self._dataLabeled[t, 0:3] for t in idx][0]
            points = np.asarray(points)
            self.cloudPlot.append(self._orEnv.plot3(points=points, pointsize=0.006, colors=colors[i], drawstyle=1))


    def executeInOR(self, postOpt):
        
        self._robot.SetDOFValues(self._graspConf)

        try:
            T = self._robot.HandObjTransform(self._graspPos[:3, :3], self._graspContacts[:, :3])
            self._robot.SetTransform(T)
        except:
            return False
        
        self.complyEEF()
        if self.checkForOutput():
            return True
        
        self.swapContacts([0,1])
        self._robot.SetDOFValues(self._graspConf)
        
        try:
            T = self._robot.HandObjTransform(self._graspPos[:3, :3], self._graspContacts[:, :3])
            self._robot.SetTransform(T)
        except:
            return False
        self.complyEEF()
        
        return self.checkForOutput()

    def composeGraspInfo(self, contactLabels):

        contacts = [] # a list of contact positions and normals
        for i in range(self._contactN):
            p, n = self.clusterRep(contactLabels[i])
            contacts.append(list(p) + list(n))

        self._graspContacts= np.asarray(contacts)


        code_tmp = self._handMani.encodeGrasp(self._graspContacts)
        dummy, self._graspConf  = self._handMani.predictHandConf(code_tmp)

        
        self._graspPos = self._robot.getOriTipPN(self._graspConf)
       

    def extendSolution(self, oldLabels):
        for label in oldLabels:
            label.append(np.random.randint(self._levels[len(label)]))
        s_tmp, r_tmp, o_tmp = self.evaluateGrasp(oldLabels)

        return o_tmp, oldLabels
 
    def clusterRep(self, label):
        level = len(label) - 1 # indexed from 0

        idx = np.where((self._dataLabeled[:, 6:7 + level] == label).all(axis=1))
        points = [self._dataLabeled[t, 0:3] for t in idx][0]
        normals = [self._dataLabeled[t, 3:6] for t in idx][0]
        pos = np.sum(points, axis=0) / len(idx[0])
        normal = np.sum(normals, axis=0) / len(idx[0])
        normal /= np.linalg.norm(normal)
        return pos, -normal
 
    def swapContacts(self, rows):
        frm = rows[0]
        to = rows[1]
        self._graspContacts[[frm, to],:] = self._graspContacts[[to, frm],:]
 
 
    def evaluateGrasp(self, contactLabel):

        contacts = []

        for i in range(self._contactN):
            p, n = self.clusterRep(contactLabel[i])
            contacts.append(list(p) + list(n))

        contacts = np.asarray(contacts)

        
        s_tmp = self._handMani.computeGraspQuality(self._objCOM, contacts)
        code_tmp = self._handMani.encodeGrasp(contacts)
        r_tmp, dummy = self._handMani.predictHandConf(code_tmp)
        # o_tmp = s_tmp - self._alpha * r_tmp
        # TODO: Research topic. This is kind of hack. Another objective function might be better
        o_tmp = s_tmp / (r_tmp + 0.000001)
        return s_tmp, r_tmp, o_tmp

    def shcEvaluation(self, o_tmp, bestO):
        if bestO < o_tmp:
            return True
        else:
            return False

        v = (bestO - o_tmp) / self._ita
        # if v < 0: #python overflow
        #     return True
        # else:
        #     return False

        p = 1. / (1 + exp(v))

        return  p > np.random.uniform()


    def getSiblingLabel(self, label):
        if len(label) <= self._hops / 2:
            ret = []
            for i in range(len(label)):
                ret.append(np.random.randint(self._levels[i]))
        else:
            matchLen = len(label) - self._hops / 2
            ret = label[:matchLen]
            for i in range(len(label) - matchLen):
                ret.append(np.random.randint(self._levels[i + matchLen]))
        return ret
 
 
    def getSiblingLabels(self, currLabels):

        labels_tmp = []
        for i in range(self._contactN):
            tmp = []
            # while tmp in labels_tmp or len(tmp) == 0:
            while len(tmp) == 0:
                tmp = self.getSiblingLabel(currLabels[i])
            labels_tmp.append(tmp)
        return labels_tmp

    def getMaximumDepth(self):
        return self._nLevel

    def setMaxIter(self, m):
        assert m > 0
        self._maxIters = m
    
    
    def complyEEF(self):
        currConf = self._robot.GetDOFValues()
        
        for i in range(100):
            currConf[1] += 0.01
            self._robot.SetDOFValues(currConf)
            if self.goodFingertipContacts():
                break
    
    def checkForOutput(self):
        # Could add more checks
        if self._robot.CheckSelfCollision():
            return False
        return self.noCollision()
    
    def goodFingertipContacts(self):
        links = self._robot.getFingertipLinks()
        
        for link in links:
            if not self._orEnv.CheckCollision(self._robot.GetLink(link)):
                return False
        return True
    
    def noCollision(self):
        
        links = self._robot.getNonFingertipLinks()
        
        
        for link in links:
            if self._orEnv.CheckCollision(self._robot.GetLink(link)):
                return False
        return True

class HFTSNode:
    def __init__(self, labels = None, handConf = None, handTransform = None,
                objTransform = None, isGoal = False):
        # None values represent the root node

        if labels is None:
            self._depth = 0
        else:
            self._depth = len(labels[0])

        self._labels = labels
        self._handConfig = handConf
        self._handTransform = handTransform
        self._objTransform = objTransform
        self._isGoal = isGoal
        
    def getLabels(self):
        return self._labels

    def getDepth(self):
        return self._depth

    def getHandConfig(self):
        return self._handConfig

    def isGoal(self):
        return self._isGoal
    
    def gethandTransform(self):
        return self._handTransform
        
    def getObjTransform(self):
        return self._objTransform



