#! /usr/bin/python


import rospy
import rospkg

if __name__ == "__main__":
    
    rospy.init_node('testPlanner')
    rospack = rospkg.RosPack()
    packPath = rospack.get_path('hfts_grasp_planner')
    dataPath = packPath + rospy.get_param('/testObj.py')
    
    rospy.spin()