#! /usr/bin/python

from plyfile import PlyData
import numpy as np
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs.msg
import rospy
from sklearn.cluster import KMeans as KMeans
import math, copy, os, itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

class ObjectFileIO:

    def __init__(self, dataPath, objectIdentifier, varFilter = True):
        self._objId = objectIdentifier
        self._dataPath = dataPath
        self._HFTSFile = self._dataPath + '/' + self._objId + '/hfts.npy'
        self._HFTSParamFile = self._dataPath + '/' + self._objId + '/hftsParam.npy'
        self._objCOMFile = self._dataPath + '/' + self._objId + '/objCOM.npy'
        self._HFTS = None
        self._HFTSParam = None
        self._objCOM = None
        self._varFilter = varFilter
        self._objFileExt = None
        
        
    def filterPoints(self, points):
        
        kdt = KDTree(points[:, :3], leaf_size = 6, metric = 'euclidean')
        rospy.loginfo('Filtering points for constructing HFTS')
        
        vldIdx = np.ones(points.shape[0], dtype=bool)
        i = 0
        for p in points:
            nbIdx = kdt.query([p[:3]], k=20, return_distance=False)[0]
            nbPointsNormals = points[nbIdx, 3:]
            var = np.var(nbPointsNormals, axis = 0)
            if max(var) > 0.2:
                vldIdx[i] = False
            i += 1
            
        points = points[vldIdx, :]
        return points
        
    
    
    
    def getPoints(self):
        objFile = self._dataPath + '/' + self._objId + '/objectModel'

        try:
            print objFile + '.ply'
            points = readPlyFile(objFile + '.ply')
            if self._varFilter:
                points = self.filterPoints(points)
                self._objFileExt = '.ply'
            return points
        except:
            rospy.loginfo('[objectFileIO] No valid \".ply\" file found for the object: ' + self._objId)
        
        try:
            points = readStlFile(objFile + '.stl')
            if self._varFilter:
                points = self.filterPoints(points)
                self._objFileExt = '.stl'
            return points
        except:
            rospy.loginfo('[objectFileIO] No valid \".stl\" file found for the object: ' + self._objId)
        
        rospy.logwarn('No previous file found in the database, will proceed with raw point cloud instead.')
        return None
    
    
        
    def getObjFileExtension(self):
        if self._objFileExt is not None:
            return self._objFileExt
        
        objFile = self._dataPath + '/' + self._objId + '/objectModel'
        
        try:
            points = os.path.isfile(objFile + '.ply')
            self._objFileExt = '.ply'

        except:
            rospy.loginfo('[objectFileIO] No \".ply\" file found for the object: ' + self._objId)

        if self._objFileExt is None:
            try:
                points = os.path.isfile(objFile + '.stl')
                self._objFileExt = '.stl'
            except:
                rospy.loginfo('[objectFileIO] No \".stl\" file found for the object: ' + self._objId)

        return self._objFileExt
        
        
        
    def getHFTS(self, forceNew = False):
        
        if self._HFTS is None or self._HFTSParam is None:

            if os.path.isfile(self._HFTSFile) and not forceNew:
                self._HFTS = np.load(self._HFTSFile)
                self._HFTSParam = np.load(self._HFTSParamFile)
                self._objCOM = np.load(self._objCOMFile)
            else:
                if not forceNew:
                    rospy.logwarn('HFTS is not available in the database')
                points = self.getPoints()
                HFTSGen = HFTSGenerator(points)
                HFTSGen.run()
                self._HFTS = HFTSGen.getHFTS()
        
                self._HFTSParam = HFTSGen.getHFTSParam()
                HFTSGen.saveHFTS(HFTSFile = self._HFTSFile, HFTSParamFile = self._HFTSParamFile, COMFile = self._objCOMFile)

        return self._HFTS, self._HFTSParam.astype(int)
    
    def getObjCOM(self):
        if self._objCOM is None:
            points = self.getPoints()
            return np.mean(points[:, :3], axis = 0)
        return self._objCOM

    
    def showHFTS(self, level):
        # This function is only for debugging purpose, will be removed
        if self._HFTS is None:
            self.getHFTS()
        
        if level > len(self._HFTSParam) - 1:
            raise ValueError('[objectFileIO::showHFTS] level ' + str(level) + ' does not exist')
            
        bFactors =  []
        for i in range(level + 1):
            bFactors.append(np.arange(self._HFTSParam[i]))
        labels = itertools.product(*bFactors)
        
        HFTSLabels = self._HFTS[:, 6:7 + level]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for label in labels:
            idx = np.where((HFTSLabels == label).all(axis=1))[0]
            clusterPoints = self._HFTS[idx, :3]
            ax.scatter(clusterPoints[:, 0], clusterPoints[:, 1], clusterPoints[:, 2], c=np.random.rand(3,1), s = 100)
        
        plt.show()
    
        
    
    
    


class HFTSGenerator:
    # 6 dim of positions and normals + labels
    def __init__(self, points):
        self._pointN = points.shape[0]
        self._objCOM = np.mean(points[:, :3], axis = 0)
        self._points = np.c_[np.arange(self._pointN), points]
        self._posWeight = 200
        
        self._branchFactor = 4
        self._firstLevelFactor = 3
        self._levelN = None
        self._HFTS = None
        self._HFTSParam = None
        

    def setPositionWeight(self, w):
        self._posWeight = w
        
    def setBranchFactor(self, b):
        self._branchFactor = b
    
    def _calLevels(self):
        self._levelN = int(math.log(self._pointN / self._firstLevelFactor, self._branchFactor)) - 1


    def _getPartitionLabels(self, points, branchFactor):
        
        points = copy.deepcopy(points)
        
        
        if points.shape[0] < branchFactor:
            self._stop = True
            rospy.loginfo('HFTS generation finished')
            return None

        estimator = KMeans(n_clusters = branchFactor)
        points[:, :3] *= self._posWeight
        estimator.fit(points)
        
        return estimator.labels_
    
    
    def _computeHFTS(self, currPoints, level = 0):

        if level >= self._levelN:
            return
        
        
        idx = currPoints[:, 0].astype(int)
        
        if level == 0:
            bFactor = self._branchFactor * self._firstLevelFactor
        else:
            bFactor = self._branchFactor
        
        points6D = currPoints[:, 1:]
        currLabels = self._getPartitionLabels(points6D, bFactor)
        
        if currLabels is None:
            self._levelN = level - 1
            return
        
        self._HFTS[idx, level] = currLabels
        
        for label in range(bFactor):
            
            lIdx = np.where(currLabels == label)[0]
            subPoints = currPoints[lIdx, :]
            self._computeHFTS(subPoints, level + 1)
        
    
    def run(self):
        if self._HFTS is not None:
            return
            
        rospy.loginfo('Generating HFTS')
        
        if self._levelN is None:
            self._calLevels()
        
        self._HFTS = np.empty([self._pointN, self._levelN])
        
        self._computeHFTS(self._points)
        
        self._HFTS = self._HFTS[:, :self._levelN]
        self._HFTSParam = np.empty(self._levelN)
        
        for i in range(self._levelN):
            if i == 0:
                self._HFTSParam[i] = self._branchFactor * self._firstLevelFactor
            else:
                self._HFTSParam[i] = self._branchFactor
        
    
        
        
            
    def saveHFTS(self, HFTSFile, HFTSParamFile, COMFile):
        data = np.c_ [self._points[:, 1:], self._HFTS]
        np.save(file = HFTSFile, arr = data)
        np.save(file = HFTSParamFile, arr = self._HFTSParam)
        np.save(file = COMFile, arr = self._objCOM)
        
    def getHFTS(self):
        if self._HFTS is None:
            self.run()
        return np.c_ [self._points[:, 1:], self._HFTS]
        
        
    def getHFTSParam(self):
        if self._HFTSParam is None:
            self.run()
        return self._HFTSParam




def readPlyFile(fileID):
    plydata = PlyData.read(fileID)
    vertex = plydata['vertex']
    (x, y, z, nx, ny, nz) = (vertex[t] for t in ('x', 'y', 'z', 'nx', 'ny', 'nz'))
    
    points = zip(x, y, z, nx, ny, nz)
    return np.asarray(points)
    
def createPointCloud(points):
    pointCloud = PointCloud()
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'
    pointCloud.header = header
    
    for point in points:
        pointCloud.points.append(Point32(point[0], point[1], point[2]))
    
    return pointCloud
    

def vecAngelDiff(v0, v1):
    # in radians
    assert len(v0) == len(v1)
    l0 = math.sqrt(np.inner(v0, v0))
    l1 = math.sqrt(np.inner(v1, v1))
    if l0 == 0 or l1 == 0:
        return 0
    x = np.dot(v0, v1) / (l0*l1)
    x = min(1.0, max(-1.0, x)) # fixing math precision error
    angel = math.acos(x)
    return angel
    
