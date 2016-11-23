#! /usr/bin/python

from plyfile import PlyData
import numpy as np
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs.msg
import rospy
from sklearn.cluster import KMeans as KMeans
import math, copy


class objectFileIO:

    def __init__(self, dataPath, objectIdentifier):
        self._objId = objectIdentifier
        self._dataPath = dataPath
        
        
        
    def getPoints(self):
        objFile = self._dataPath + '/' + self._objId + '/' + self._objId
        
        try:
            points = readPlyFile(objFile + '.ply')
            return points
        except:
            rospy.loginfo('[objectFileIO] No \".ply\" file found for the object: ' + self._objId)
        
        try:
            points = readStlFile(objFile + '.stl')
            return points
        except:
            rospy.loginfo('[objectFileIO] No \".stl\" file found for the object: ' + self._objId)
        
        rospy.logwarn('No previous file found in the database, will proceed with raw point cloud instead.')
        return None
    
    def getHFTS(self):
        # Check if available in the database
        pass


        
    
    
        
    
    
    


class HFTSGenerator:
    # 6 dim of positions and normals + labels
    def __init__(self, points):
        self._pointN = points.shape[0]
        self._points = np.c_[np.arange(self._pointN), points]
        self._posWeight = 20
        self._branchFactor = 4
        self._firstLevelFactor = 5
        self._levelN = None
        self._HFTS = None

    def setPositionWeight(self, w):
        self._posWeight = w
        
    def setBranchFactor(self, b):
        self._branchFactor = b
    
    def _calLevels(self):
        self._levelN = math.log(self._pointN / self._firstLevelFactor, self._branchFactor)

    
    def _getPartitionLabels(self, points, branchFactor):

        estimator = KMeans(n_clusters = branchFactor)
        points[:, 3:] *= self._posWeight
        estimator.fit(points[:, 1:])
        
        return estimator.labels_
    
    
    

    
    def _computeHFTS(self, currPoints, level = 0):

        if level == self._levelN:
            return
        
        idx = currPoints[:, 0]
        
        if level == 0:
            bFactor = self._branchFactor * self._firstLevelFactor
        else:
            bFactor = self._branchFactor
        
        points6D = currPoints[:, 1:]
        currLabels = self._getPartitionLabels(points6D, bFactor)
        self._HFTS[idx, level] = currLabels
        
        for label in range(bFactor):
            lIdx = np.where((currLabels == label).all(axis = 1))
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
            
        
    def getHFTS(self):
        return np.c_ [self._points, self._HFTS]
        
        





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
    
    
    
    
    
