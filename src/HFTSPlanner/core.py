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

    def __init__(self, verbose=False, numHops=2, vis=True):

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
        print self._levels
        self._nLevel = len(self._levels)
        self._obj = self._orEnv.Load(dataPath + '/' + objId + '/objectModel' + objectIO.getObjFileExtension())
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
        self._armConf = None
        self._handPose_lab = None
        shift = transformations.identity_matrix()
        shift[0,-1] = 0.2
        self._robot.SetTransform(shift)
        # self._robot.SetDOFValues([0]*7, range(7))
        self.handles = []
        self.tipPNHandler = []



#     def checkGraspIK(self, seed=None, openHandOffset=0.1):
#         # this function can be called after composeGraspInfo is called
#         objPose_lab = self._orLabEnv.getObjTransform(self._objName)
#         objPose = self._obj.GetTransform()
#         handPose_hfts = np.dot(np.linalg.inv(objPose), self._graspPose)
#         handPose_lab = np.dot(objPose_lab, handPose_hfts)
#         bValid, armConf, self._preGraspConf = self._orLabEnv.handCheckIK6D(handPose_lab, self._graspConf,
#                                                                            seed=seed,
#                                                                            openHandOffset=openHandOffset)
# 
#         self._armConf = armConf
#         self._handPose_lab = handPose_lab
#         return bValid
# 
# 
    def sampleGrasp(self, node, depthLimit, postOpt=False, openHandOffset=0.1):

        assert depthLimit >= 0
        if node.getDepth() >= self._nLevel:
            raise ValueError('graspSampler::sampleGrasp input node has an invalid depth')
            
            
        if node.getDepth() + depthLimit > self._nLevel:
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
                self.executeInOR(postOpt = postOpt)
                return

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

        # initial alignment
        T = self._robot.HandObjTransform(self._graspPos[:3, :3], self._graspContacts[:, :3])
        self._robot.SetTransform(T)
        # raw_input('press')
        return
        if postOpt: # postOpt
            rot = transformations.rotation_from_matrix(T)
            # further optimize hand configuration
            rotParam = rot[1].tolist() + [rot[0]] + T[:3, -1].tolist()
            fmin_cobyla(self._robot.allObj, self._robot.GetDOFValues() + rotParam, allConstr, rhobeg = .1,
                        rhoend=1e-4, args=(self._graspContacts[:, :3], self._graspContacts[:, 3:], self._robot), maxfun=1e8, iprint=0)
        # 
        # self.complyEndEffectors()
        # 
        # self._graspPose = self._robot.GetTransform()
        # self._graspConf = self._robot.GetDOFValues()
        # # self.drawTipPN()
        # ret, stability = self.finalCheck()
        # return ret, stability
        
        # if ret < 3:
        #     return ret
        # else:
        #     self.swapContacts([0,2])
        # code_tmp = self._handMani.encodeGrasp3(self._graspContacts)
        # dummy, self._graspConf  = self._handMani.predictHandConf(code_tmp)
        # self._robot.SetDOFValues(self._graspConf, range(self._robot.getHandDim()[1]))
        # self._graspPos = self._handMani.getOriTipPN(self._graspConf)
        #
        # # initial alignment
        # T = self._robot.HandObjTransform(self._graspPos[:3, :3], self._graspContacts[:, :3])
        # self._robot.SetTransform(T)
        #
        #
        # if postOpt: # postOpt
        #     rot = transformations.rotation_from_matrix(T)
        #     # further optimize hand configuration
        #     rotParam = rot[1].tolist() + [rot[0]] + T[:3, -1].tolist()
        #     fmin_cobyla(self._robot.allObj, self._robot.GetDOFValues() + rotParam, allConstr, rhobeg = .1,
        #                 rhoend=1e-4, args=(self._graspContacts[:, :3], self._graspContacts[:, 3:], self._robot), maxfun=1e8, iprint=0)
        #
        # self.complyEndEffectors()
        #
        # self._graspPose = self._robot.GetTransform()
        # self._graspConf = self._robot.GetDOFValues()
        # # self.drawTipPN()
        # return self.finalCheck()
        #
# 
# 
#     def finalCheck(self):
#         stability = -1.0
#         if self._robot.CheckSelfCollision():
#             return 5, stability
#         if self.checkContacts():
#             return 4, stability
#         if self.checkCollision():
#             return 3, stability
#         stability = self.computeRealStability()
#         if stability < 0.001:
#             # print 'real stability: %f' % self.computeRealStability()
#             return 2, stability
#         if self.computeContactQ() > 60:
#             return 1, stability
# 
#         return 0, stability

    def checkFingertipContacts(self):
        pass
# 
#     def checkContacts(self):
#         links = self._robot.GetLinks()
#         for link in self._robot.getEndEffectors():
#             if not self._orEnv.CheckCollision(links[link], self._obj):
#                 return True
#         return False
# 
# 
#     def computeRealStability(self):
# 
#         try:
#             q = hfts_utils.computeGraspQualityNeg(self.getRealContacts(), self._mu)
#             return q
#         except:
#             return -1.
# 
# 
#     def computeContactQ(self):
#         try:
#             rc = self.getRealContacts()
#         except:
#             return np.inf
#         tipPN = self._robot.getTipPN()
#         # ret = self._robot.contactsDiff(-rc[:,3:], tipPN[:,3:])
#         ret = 0
#         for i in range(self._robot.getHandDim()[-1]):
#             ret = max(ret, hfts_utils.vecAngelDiff(rc[i, 3:], -tipPN[i, 3:]))
#         return ret
# 
#     def getRealContacts(self):
#         reportX = orpy.CollisionReport()
#         rContacts = []
# 
#         for eel in self._robot.getEndEffectors():
# 
#             self._orEnv.CheckCollision(self._obj, self._robot.GetLinks()[eel],report=reportX)
#             if len(reportX.contacts) == 0:
#                 raise ValueError('no contact found')
#             rContacts.append(np.concatenate((reportX.contacts[0].pos, reportX.contacts[0].norm)))
# 
#         rContacts = np.asarray(rContacts)
#         return rContacts
# 
# 
#     def checkCollision(self):
#         i = -1
# 
#         for link in self._robot.GetLinks():
#             i += 1
# 
#             if i in self._robot.getEndEffectors():
#                 continue
#             if self._orEnv.CheckCollision(link, self._obj):
#                 return True
#         return False
# 
# 
#     def complyEndEffectors(self):
# 
#         curr = self._robot.GetDOFValues()
#         for j in self._robot.getEndJoints():
#             curr[j] = -math.pi/2.
#             # curr[j-1] -=math.pi/36
# 
# 
#         limitL, limitU = self._robot.GetDOFLimits()
#         self._robot.SetDOFValues(np.asarray(curr), range(self._robot.getHandDim()[1]))
# 
#         stepLen = 0.2
# 
#         joints = self._robot.getEndJoints()
#         links = self._robot.getEndEffectors()
# 
# 
# 
#         for i in range(len(joints)):
# 
#             maxStep = 400
#             while self._orEnv.CheckCollision(self._robot.GetLinks()[links[i]-1]):
#                 curr[joints[i]-1] -= stepLen
#                 if curr[joints[i]-1] < limitL[joints[i]-1]:
#                     break
# 
#                 self._robot.SetDOFValues(curr, range(self._robot.getHandDim()[1]))
# 
#         stepLen = 0.01
#         maxStep = 400
#         done = [False] * len(joints)
# 
# 
#         while False in done and maxStep >= 0:
#             curr = self._robot.GetDOFValues()
#             for i in range(len(joints)):
#                 if curr[joints[i]] >= limitU[joints[i]]:
#                     done[i] = True
#                 if not done[i]:
#                     curr[joints[i]] += stepLen
#                     if self._orEnv.CheckCollision(self._robot.GetLinks()[links[i]]):
#                         done[i] = True
#                         curr[joints[i]] += stepLen
# 
#             maxStep -= 1
# 
#             self._robot.SetDOFValues(curr, range(self._robot.getHandDim()[1]))
# 
#     def plotContacts(self, cPoints, clear=False):
#         if not self._samplerViewer:
#             return
#         pointSize = 0.008
#         if clear:
#             self.handles = []
#         colors = [np.array((1,0,0)), np.array((1,1,0)), np.array((1,0,0))]
#         c0 = cPoints[0, :3]
#         c1 = cPoints[1, :3]
#         c2 = cPoints[2, :3]
# 
#         n0 = cPoints[0, 3:]
#         n1 = cPoints[1, 3:]
#         n2 = cPoints[2, 3:]
# 
# 
#         self.handles.append(self._orEnv.plot3(points=c0, pointsize=pointSize, colors=colors[0],drawstyle=1))
#         self.handles.append(self._orEnv.plot3(points=c1, pointsize=pointSize, colors=colors[1],drawstyle=1))
#         self.handles.append(self._orEnv.plot3(points=c2, pointsize=pointSize, colors=colors[2],drawstyle=1))
# 
#         self.handles.append(self._orEnv.drawarrow(p1=c0, p2=c0 + 0.02 * n0,linewidth=0.001,color=colors[0]))
#         self.handles.append(self._orEnv.drawarrow(p1=c1, p2=c1 + 0.02 * n1,linewidth=0.001,color=colors[1]))
#         self.handles.append(self._orEnv.drawarrow(p1=c2, p2=c2 + 0.02 * n2,linewidth=0.001,color=colors[2]))
# 
# 
# 
# 
    def composeGraspInfo(self, contactLabels):

        contacts = [] # a list of contact positions and normals
        for i in range(self._contactN):
            p, n = self.clusterRep(contactLabels[i])
            contacts.append(list(p) + list(n))

        self._graspContacts= np.asarray(contacts)


        code_tmp = self._handMani.encodeGrasp(self._graspContacts)
        dummy, self._graspConf  = self._handMani.predictHandConf(code_tmp)

        
        self._graspPos = self._robot.getOriTipPN(self._graspConf)
#       
# 
    def extendSolution(self, oldLabels):
        for label in oldLabels:
            label.append(np.random.randint(self._levels[len(label)]))
        s_tmp, r_tmp, o_tmp = self.evaluateGrasp(oldLabels)

        return o_tmp, oldLabels
# 
    def clusterRep(self, label):
        level = len(label) - 1 # indexed from 0

        idx = np.where((self._dataLabeled[:, 6:7 + level] == label).all(axis=1))
        points = [self._dataLabeled[t, 0:3] for t in idx][0]
        normals = [self._dataLabeled[t, 3:6] for t in idx][0]
        pos = np.sum(points, axis=0) / len(idx[0])
        normal = np.sum(normals, axis=0) / len(idx[0])
        normal /= np.linalg.norm(normal)
        return pos, -normal
# 
#     def swapContacts(self, rows):
#         frm = rows[0]
#         to = rows[1]
#         self._graspContacts[[frm, to],:] = self._graspContacts[[to, frm],:]
# 
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
# 
# 
# 
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
# 
# 
#     def drawTipPN(self):
# 
#         if not self._samplerViewer:
#             return
#         self.tipPNHandler = []
# 
#         tipPN = self._robot.getTipPN()
#         pointSize = 0.008
# 
#         colors = [np.array((1,0,1)), np.array((1,0,1)), np.array((1,0,1))]
#         c0 = tipPN[0, :3]
#         c1 = tipPN[1, :3]
#         c2 = tipPN[2, :3]
# 
#         n0 = tipPN[0, 3:]
#         n1 = tipPN[1, 3:]
#         n2 = tipPN[2, 3:]
# 
# 
#         self.tipPNHandler.append(self._orEnv.plot3(points=c0, pointsize=pointSize, colors=colors[0],drawstyle=1))
#         self.tipPNHandler.append(self._orEnv.plot3(points=c1, pointsize=pointSize, colors=colors[1],drawstyle=1))
#         self.tipPNHandler.append(self._orEnv.plot3(points=c2, pointsize=pointSize, colors=colors[2],drawstyle=1))
# 
#         self.tipPNHandler.append(self._orEnv.drawarrow(p1=c0, p2=c0 + 0.02 * n0,linewidth=0.001,color=colors[0]))
#         self.tipPNHandler.append(self._orEnv.drawarrow(p1=c1, p2=c1 + 0.02 * n1,linewidth=0.001,color=colors[1]))
#         self.tipPNHandler.append(self._orEnv.drawarrow(p1=c2, p2=c2 + 0.02 * n2,linewidth=0.001,color=colors[2]))
# 
# 
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
# 
# 
    def getSiblingLabels(self, currLabels):

        labels_tmp = []
        for i in range(self._contactN):
            tmp = []
            # while tmp in labels_tmp or len(tmp) == 0:
            while len(tmp) == 0:
                tmp = self.getSiblingLabel(currLabels[i])
            labels_tmp.append(tmp)
        return labels_tmp
# 
#     def getMaximumDepth(self):
#         return self._nLevel
# 
#     def setAlpha(self, a):
#         assert a > 0
#         self._alpha = a
# 
#     def setMaxIter(self, m):
#         assert m > 0
#         self._maxIters = m
# 
#     def getRootNode(self):
#         possibleNumChildren, possibleNumLeaves = self.getBranchInformation(0)
#         return HFTSNode(possibleNumChildren=possibleNumChildren,
#                         possibleNumLeaves=possibleNumLeaves)
# 
class HFTSNode:
    def __init__(self, labels=None, handConf=None):
        # None values represent the root node

        if labels is None:
            self._depth = 0
        else:
            self._depth = len(labels[0])

        self._labels = labels
        self._handConfig = handConf

    def getLabels(self):
        return self._labels

    def getUniqueLabel(self):
        if self._labels is None:
            return 'root'
        label = []
        for fingerLabel in self._labels:
            label.extend(fingerLabel)
        return str(label)

    def isExtendible(self):
        return not self._bIsLeaf

    def getContactLabels(self):
        return self._labels

    def isLeaf(self):
        return self._bIsLeaf

    def getDepth(self):
        return self._depth

    def getHandConfig(self):
        return self._handConfig

    def getPreGraspHandConfig(self):
        return self._preGraspHandConfig

    def getArmConfig(self):
        return self._armConfig

    def getPossibleNumChildren(self):
        return self._possibleNumChildren

    def getPossibleNumLeaves(self):
        return self._possibleNumLeaves

    def isGoal(self):
        return self._goal

    def isValid(self):
        return self._valid

    def getQuality(self):
        return self._quality

    def hasConfiguration(self):
        return self._armConfig is not None and self._handConfig is not None

