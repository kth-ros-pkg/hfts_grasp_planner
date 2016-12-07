#! /usr/bin/python

import rospy
import rospkg
from HFTSPlanner.utils import *
from HFTSPlanner.core import graspSampler
from HFTSPlanner.core import HFTSNode
from hfts_grasp_planner.srv import PlanGrasp, PlanGraspRequest, PlanGraspResponse
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
import tf.transformations as tff
from std_msgs.msg import Header

class handlerClass(object):
    def __init__(self):
        rospack = rospkg.RosPack()
        self._packPath = rospack.get_path('hfts_grasp_planner')
        handFile = self._packPath + rospy.get_param('/handFile')

        self._planner = graspSampler()
        self._planner.loadHand(handFile)
        self._plan = None
        
    def handle_plan_request(self, req):
        
    #    pointCloud = req.point_cloud

        
        rootHFTSNode = HFTSNode()
        orHand = self._planner.getOrHand()
        self._planner.loadObj(self._packPath + '/data', req.object_identifier)
        
        

    
        while not rospy.is_shutdown():
            
            retNode = self._planner.sampleGrasp(rootHFTSNode, 30)
            if retNode.isGoal():
                graspPose = retNode.gethandTransform()
                graspPoseQuaternion = tff.quaternion_from_matrix(graspPose)
                graspPosition = graspPose[:3, -1]
                
                # responseGraspPose = Pose()
                # responseGraspPose.position.x = graspPosition[0]
                # responseGraspPose.position.y = graspPosition[1]
                # responseGraspPose.position.z = graspPosition[2]
                # responseGraspPose.orientation.x = graspPoseQuaternion[0]
                # responseGraspPose.orientation.y = graspPoseQuaternion[1]
                # responseGraspPose.orientation.z = graspPoseQuaternion[2]
                # responseGraspPose.orientation.w = graspPoseQuaternion[3]
                # 
                # 
                # header = Header()
                # header.frame_id = req.object_identifier
                # header.stamp = rospy.Time.now()
                # 
                # 
                # stampedGraspPose = PoseStamped()
                # stampedGraspPose.pose = responseGraspPose
                # stampedGraspPose.header = header
                # 
                # handConf = retNode.getHandConfig()
                # responseHandConf = JointState()
                # responseHandConf.header = header
                # responseHandConf.position = handConf
                # 
                # joints = orHand.GetJoints()
                # for joint in joints:
                #     responseHandConf.name.append(joint.GetName())
                # 
        #         return PlanGraspResponse(True, stampedGraspPose, responseHandConf)
                return
        

if __name__ == "__main__":
    
    rospy.init_node('hfts_planner_node')
    handler = handlerClass()
    s = rospy.Service('plan_grasp', PlanGrasp, handler.handle_plan_request)
    rospy.spin()


