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

class objectFileIO:

    def __init__(self, dataPath, objectIdentifier, varFilter = True):
        self._objId = objectIdentifier
        self._dataPath = dataPath
        self._HFTSFile = self._dataPath + '/' + self._objId + '/' + self._objId + '_hfts.npy'
        self._HFTSParamFile = self._dataPath + '/' + self._objId + '/' + self._objId + '_hftsParam.npy'
        self._HFTS = None
        self._HFTSParam = None
        self._varFilter = varFilter
        
        
    def filterPoints(self, points):
        
        rospy.loginfo('Filtering points for constructing HFTS')
        kdt = KDTree(points[:, :3], leaf_size = 6, metric = 'euclidean')
        
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
        objFile = self._dataPath + '/' + self._objId + '/' + self._objId
        
        try:
            points = readPlyFile(objFile + '.ply')
            if self._varFilter:
                points = self.filterPoints(points)
            return points
        except:
            rospy.loginfo('[objectFileIO] No \".ply\" file found for the object: ' + self._objId)
        
        try:
            points = readStlFile(objFile + '.stl')
            if self._varFilter:
                points = self.filterPoints(points)
            return points
        except:
            rospy.loginfo('[objectFileIO] No \".stl\" file found for the object: ' + self._objId)
        
        rospy.logwarn('No previous file found in the database, will proceed with raw point cloud instead.')
        return None
    
    def getHFTS(self):
        
        if self._HFTS is None or self._HFTSParam is None:

            if os.path.isfile(self._HFTSFile):
                self._HFTS = np.load(self._HFTSFile)
                self._HFTSParam = np.load(self._HFTSParamFile)
         
            else:
                rospy.logwarn('HFTS is not available in the database')
                points = self.getPoints()
                HFTSGen = HFTSGenerator(points)
                HFTSGen.run()
                self._HFTS = HFTSGen.getHFTS()
                self._HFTSParam = HFTSGen.getHFTSParam()
                HFTSGen.saveHFTS(HFTSFile = self._HFTSFile, HFTSParamFile = self._HFTSParamFile)

        return self._HFTS, self._HFTSParam

    
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
        points = self._HFTS[:, :3] * 0.99

        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='white', s = 200)

        for label in labels:
            idx = np.where((HFTSLabels == label).all(axis=1))[0]
            clusterPoints = self._HFTS[idx, :3]
            ax.scatter(clusterPoints[:, 0], clusterPoints[:, 1], clusterPoints[:, 2], c=np.random.rand(3,1), s = 100)
        
        plt.show()
    
        
    
    
    


class HFTSGenerator:
    # 6 dim of positions and normals + labels
    def __init__(self, points):
        self._pointN = points.shape[0]
        self._points = np.c_[np.arange(self._pointN), points]
        self._posWeight = 20
        self._branchFactor = 2
        self._firstLevelFactor = 4
        self._levelN = None
        self._HFTS = None
        self._HFTSParam = None

    def setPositionWeight(self, w):
        self._posWeight = w
        
    def setBranchFactor(self, b):
        self._branchFactor = b
    
    def _calLevels(self):
        self._levelN = int(math.log(self._pointN / self._firstLevelFactor, self._branchFactor)) - 2
        
    

    def _getPartitionLabels(self, points, branchFactor):

        estimator = KMeans(n_clusters = branchFactor)
        points[:, :3] *= self._posWeight
        estimator.fit(points)
        
        return estimator.labels_
    
    
    def _computeHFTS(self, currPoints, level = 0):

        if level == self._levelN:
            return
        
        idx = currPoints[:, 0].astype(int)
        
        if level == 0:
            bFactor = self._branchFactor * self._firstLevelFactor
        else:
            bFactor = self._branchFactor
        
        points6D = currPoints[:, 1:]
        currLabels = self._getPartitionLabels(points6D, bFactor)

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
        
        self._HFTSParam = np.empty(self._levelN)
        
        for i in range(self._levelN):
            if i == 0:
                self._HFTSParam[i] = self._branchFactor * self._firstLevelFactor
            else:
                self._HFTSParam[i] = self._branchFactor
        
    
        
        
            
    def saveHFTS(self, HFTSFile, HFTSParamFile):
        data = np.c_ [self._points[:, 1:], self._HFTS]
        np.save(file = HFTSFile, arr = data, allow_pickle = False)
        np.save(file = HFTSParamFile, arr = self._HFTSParam, allow_pickle = False)


        
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
    
    

    
    
