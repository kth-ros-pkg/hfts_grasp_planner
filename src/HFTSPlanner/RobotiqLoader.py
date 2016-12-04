

import numpy as np
import transformations
import math
from utils import vecAngelDiff

class RobotiqHand:

    def __init__(self, env=None, handFile=None):

        self._orEnv = env
        self._orEnv.Load(handFile)
        self._orHand = self._orEnv.GetRobots()[0]
        self._handMain = RobotiqHandVirtualMainfold(self._orHand)
        
    def getHandMani(self):
        return self._handMain
    
    def plotFingertipContacts(self):
        self._plotHandler = []
        colors = [np.array((1,0,0)), np.array((0,1,0)), np.array((0,0,1))]
        tipLinkIds = [5, 9, 13]
        pointSize = 0.005
        links = self._orHand.GetLinks()
        
        for i in range(len(tipLinkIds)):
            link = links[tipLinkIds[i]]
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

        tipLinkIds = [5, 9, 13]
        links = self._orHand.GetLinks()
        
        
        ret = []
        for i in range(len(tipLinkIds)):
            link = links[tipLinkIds[i]]
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
            

    
class RobotiqHandVirtualMainfold:
    
    '''
    Mimic the hand minifold interface from our ICRA'16 paper, it is not needed to model a reachability manifold for the Robotiq-S.
    '''
    def __init__(self, orHand):

        self._orHand = orHand
    



    def predictHandConf(self, q):
        # Not accurate yet, to be fixed!!!!!!!!!!!!!!
        
        diff01Range = [0.026, 0.122]
        diff2cRange = [0, 0.165]
        dofLimits = self._orHand.GetDOFLimits()
        
            
    
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
        aDiff2A = vecAngelDiff(grasp[2, 3:], avgN01)
        
        return [d01, d2c, aDiff01, aDiff2A]

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    