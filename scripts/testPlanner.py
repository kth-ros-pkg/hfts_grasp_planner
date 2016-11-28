#! /usr/bin/python


import rospy
import rospkg
from HFTSPlanner.utils import *

if __name__ == "__main__":
    
    rospy.init_node('testPlanner')
    rospack = rospkg.RosPack()
    packPath = rospack.get_path('hfts_grasp_planner')
    objFile = rospy.get_param('/testObj')
    
    objectIO = objectFileIO(packPath + '/data', objFile)
    objPoints = objectIO.getPoints()
    HFTS, HFTSParam = objectIO.getHFTS()
    print HFTSParam
    
    objectIO.showHFTS(4)
    # while not rospy.is_shutdown():
    #     rospy.sleep(1)
    # objPointCloud = createPointCloud(objPoints)
    # pointcloud_publisher = rospy.Publisher("/testPointCloud", PointCloud, queue_size=1)
    # 
    # while not rospy.is_shutdown():
    #     pointcloud_publisher.publish(objPointCloud)
    #     rospy.sleep(0.1)