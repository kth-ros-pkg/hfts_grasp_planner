

import numpy as np
import transformations
import math
from utils import vecAngelDiff
import rospy

class RobotiqHand:

    def __init__(self, env=None, handFile=None):

        self._orEnv = env
        self._orEnv.Load(handFile)
        self._orHand = self._orEnv.GetRobots()[0]
        self._handMani = RobotiqHandVirtualMainfold(self._orHand)
    
    def __getattr__(self, attr): # composition
        return getattr(self._orHand, attr)
        
    def getHandMani(self):
        return self._handMani
    
    def plotFingertipContacts(self):
        self._plotHandler = []
        colors = [np.array((1,0,0)), np.array((0,1,0)), np.array((0,0,1))]
        tipLinkIds = ['finger_1_link_3', 'finger_2_link_3', 'finger_middle_link_3']
        pointSize = 0.005
        
        for i in range(len(tipLinkIds)):
            link = self._orHand.GetLink(tipLinkIds[i])
            T = link.GetGlobalMassFrame()
            localFrameRot = transformations.rotation_matrix(math.pi / 6., [0, 0, 1])[:3, :3]
            T[:3, :3] = T[:3, :3].dot(localFrameRot)
            
            offset = T[0:3,0:3].dot(self.getTipOffsets())
            T[0:3,3] = T[0:3,3] + offset
            
            position = T[:3, -1]
            self._plotHandler.append(self._orEnv.plot3(points=position, pointsize=pointSize, colors=colors[i],drawstyle=1))
            for j in range(3):
                normal = T[:3, j]
                self._plotHandler.append(self._orEnv.drawarrow(p1=position, p2=position + 0.05 * normal,linewidth=0.001,color=colors[j]))
    
    def getTipOffsets(self):
        return np.array([0.025, 0.006, 0.0])
        

    def getTipTransforms(self):

        tipLinkIds = ['finger_1_link_3', 'finger_2_link_3', 'finger_middle_link_3']
        
        ret = []
        
        for i in range(len(tipLinkIds)):
            link = self._orHand.GetLink(tipLinkIds[i])
            T = link.GetGlobalMassFrame()
            localFrameRot = transformations.rotation_matrix(math.pi / 6., [0, 0, 1])[:3, :3]
            T[:3, :3] = T[:3, :3].dot(localFrameRot)
            offset = T[0:3,0:3].dot(self.getTipOffsets())
            T[0:3,3] = T[0:3,3] + offset
            ret.append(T)
        return ret
    
    def getTipPN(self):
        ret = []
        tfs = self.getTipTransforms()
        for t in tfs:
            ret.append(np.concatenate((t[:3, 3], t[:3, 1])))
        
        return np.asarray(ret)
    
    def setRandomConf(self):
        lower, upper = self._orHand.GetDOFLimits()
        upper[1] = 0.93124747
        selfCollision = True
        while selfCollision:
            ret = []
            for i in range(2):
                ret.append(np.random.uniform(lower[i], upper[i]))
            self.SetDOFValues(ret)
            selfCollision = self._orHand.CheckSelfCollision()
            


    def getContactNumber(self):
        return 3

    
class RobotiqHandVirtualMainfold:
    
    '''
    Mimic the hand minifold interface from our ICRA'16 paper, it is not needed to model a reachability manifold for the Robotiq-S.
    '''
    def __init__(self, orHand):

        self._orHand = orHand
        self._alpha = 10



    def predictHandConf(self, q):

        range0 = np.array([0.0255, 0.122])
        range1 = np.array([0, 0.165])
        lower, upper = self._orHand.GetDOFLimits()
        upper[1] = 0.93124747
        
        
        posResidual0 = self.distInRange(q[0], range0)
        posResidual1 = self.distInRange(q[1], range1)
        angleResidual0 = q[2]
        angleResidual1 = q[3]
        
        res0 = (upper[0] - lower[0]) / (range0[1] - range0[0])
        res1 = (upper[1] - lower[1]) / (range1[1] - range1[0])
        
        if posResidual0 == 0:
            joint0 = lower[0] + (range0[1] - q[0]) * res0
        elif posResidual0 > 0 and q[0] < range0[0]:
            joint0 = upper[0]
        elif posResidual0 > 0 and q[0] > range0[1]:
            joint0 = lower[0]
        else:
            raise ValueError('[RobotiqHandVirtualMainfold::predictHandConf] grasp encoding is incorrect')

        if posResidual1 == 0:
            joint1 = lower[1] + (range1[1] - q[1]) * res1
        elif posResidual1 > 0 and q[1] < range1[0]:
            joint1 = upper[1]
        elif posResidual1 > 0 and q[1] > range1[1]:
            joint1 = lower[1]
        else:
            raise ValueError('[RobotiqHandVirtualMainfold::predictHandConf] grasp encoding is incorrect')
        
        return self.getPredRes(q, [range0, range1]), [joint0, joint1]
    
    def computeGraspQuality(self, objCOM, grasp):
        # need to be tested
        contacts = grasp[:, :3]
        centerShift = contacts - objCOM
        d = np.linalg.norm(centerShift)
        
        v01 = grasp[0, :3] - grasp[1, :3]
        c01 = (grasp[0, :3] + grasp[1, :3]) / 2.
        v2c = grasp[2, :3] - c01
        
        d01 = np.linalg.norm(v01)
        d2c = np.linalg.norm(v2c)
        
        return (d01 + d2c) - d * 2
        
    
    def getPredRes(self, q, ranges):
        range0, range1 = ranges
        posResidual0 = self.distInRange(q[0], range0)
        posResidual1 = self.distInRange(q[1], range1)
        angleResidual0 = q[2]
        angleResidual1 = q[3]
        
        r = (posResidual0 + posResidual1) * self._alpha + (angleResidual0 + angleResidual1)
        return r
        
        
    def distInRange(self, d, r):
        
        if d < r[0]:
            return r[0] - d
        elif d > r[1]:
            return d - r[1]
        else:
            return 0.0
    
    def encodeGrasp(self, grasp):
        v01 = grasp[0, :3] - grasp[1, :3]
        c01 = (grasp[0, :3] + grasp[1, :3]) / 2.
        v2c = grasp[2, :3] - c01
        
        d01 = np.linalg.norm(v01)
        d2c = np.linalg.norm(v2c)
        
        aDiff01 = vecAngelDiff(grasp[0, 3:], grasp[1, 3:])
        avgN01 = (grasp[0, 3:] + grasp[1, 3:]) / 2.
        aDiff2A = vecAngelDiff(grasp[2, 3:], -avgN01)
        
        return [d01, d2c, aDiff01, aDiff2A]

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    