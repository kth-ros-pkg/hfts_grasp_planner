#! /usr/bin/python


import rospy
import rospkg
from HFTSPlanner.utils import *
from HFTSPlanner.core import graspSampler
from HFTSPlanner.core import HFTSNode
from hfts_grasp_planner.srv import PlanGrasp

if __name__ == "__main__":
    
    rospy.init_node('testPlanner')
    rospack = rospkg.RosPack()
    packPath = rospack.get_path('hfts_grasp_planner')
    objFile = rospy.get_param('/testObj')
    
    # objectIO = objectFileIO(packPath + '/data', objFile)
    # objPoints = objectIO.getPoints()
    # HFTS, HFTSParam = objectIO.getHFTS(forceNew = True)

    # objectIO.showHFTS(len(HFTSParam)-1)
    # while not rospy.is_shutdown():
    #     rospy.sleep(1)
    # objPointCloud = createPointCloud(objPoints)
    # pointcloud_publisher = rospy.Publisher("/testPointCloud", PointCloud, queue_size=1)
    # 
    # while not rospy.is_shutdown():
    #     pointcloud_publisher.publish(objPointCloud)
    #     rospy.sleep(0.1)
    # 
    # planer = graspSampler()
    # 
    # handFile = packPath + rospy.get_param('/handFile')
    # 
    # planer.loadHand(handFile)
    # robot = planer._robot
    # handMani = robot.getHandMani()
    # 
    # while not rospy.is_shutdown():
    #     robot.setRandomConf()
    #     robot.plotFingertipContacts()
    #     grasp = robot.getTipPN()
    #     q = handMani.encodeGrasp(grasp)
    #     raw_input('press to predict')
    #     residual, conf = handMani.predictHandConf(q)
    #     print residual
    # 
    #     robot.SetDOFValues(conf)
    #     raw_input('press for next')
    #     
    #     
    #     rospy.sleep(0.1)
    
    rootHFTSNode = HFTSNode()
    planner = graspSampler(vis=True)
    handFile = packPath + rospy.get_param('/handFile')
    planner.load_hand(handFile)
    planner.load_object(packPath + '/data', objFile)
    finished = False
    while not finished:
        retNode = planner.sample_grasp(rootHFTSNode, 5)
        finished = retNode.is_goal()
        rospy.sleep(0.2)
    
    rospy.spin()
    
