#! /usr/bin/python

import rospy
import rospkg
from hfts_grasp_planner.utils import *
from hfts_grasp_planner.core import graspSampler, HFTSNode
from hfts_grasp_planner.srv import PlanGrasp, PlanGraspRequest, PlanGraspResponse
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
import tf.transformations as tff
from std_msgs.msg import Header


PACKAGE_NAME = 'hfts_grasp_planner'


class HandlerClass(object):
    """ Class that provides a ROS service callback for executing the HFTS grasp planner."""

    def __init__(self):
        """ Creates a new handler class."""
        rospack = rospkg.RosPack()
        self._package_path = rospack.get_path(PACKAGE_NAME)
        # Create planner
        b_visualize = rospy.get_param('visualize', default=False)
        self._planner = graspSampler(vis=b_visualize)
        # Load hand and save joint names
        hand_file = self._package_path + rospy.get_param('handFile')
        self._planner.loadHand(hand_file)
        or_hand = self._planner.getOrHand()
        joints = or_hand.GetJoints()
        self._joint_names = []
        for joint in joints:
            self._joint_names.append(joint.GetName())

    def handle_plan_request(self, req):
        """ Callback function for a grasp planning servce request. """
        # TODO generate HFTS from point cloud if point cloud is specified
        # pointCloud = req.point_cloud
        # Load the requested object first
        self._planner.loadObj(self._package_path + '/data', req.object_identifier)
        # We always start from the root node, so create a root node
        root_hfts_node = HFTSNode()
        max_iterations = rospy.get_param('max_iterations', 20)
        iteration = 0
        # Iterate until either shutdown, max_iterations reached or a good grasp was found
        while iteration < max_iterations and not rospy.is_shutdown():
            return_node = self._planner.sampleGrasp(root_hfts_node, 30)
            if return_node.isGoal():
                grasp_pose = return_node.getHandTransform()
                pose_quaternion = tff.quaternion_from_matrix(grasp_pose)
                pose_position = grasp_pose[:3, -1]
                # Save pose in ROS pose
                ros_grasp_pose = Pose()
                ros_grasp_pose.position.x = pose_position[0]
                ros_grasp_pose.position.y = pose_position[1]
                ros_grasp_pose.position.z = pose_position[2]
                ros_grasp_pose.orientation.x = pose_quaternion[0]
                ros_grasp_pose.orientation.y = pose_quaternion[1]
                ros_grasp_pose.orientation.z = pose_quaternion[2]
                ros_grasp_pose.orientation.w = pose_quaternion[3]
                # Make a header for the message
                header = Header()
                header.frame_id = req.object_identifier
                header.stamp = rospy.Time.now()
                # Create stamped pose
                stamped_ros_grasp_pose = PoseStamped()
                stamped_ros_grasp_pose.pose = ros_grasp_pose
                stamped_ros_grasp_pose.header = header
                # Create JointState message to send hand configuration
                hand_conf = return_node.getHandConfig()
                ros_hand_joint_state = JointState()
                ros_hand_joint_state.header = header
                ros_hand_joint_state.position = hand_conf
                ros_hand_joint_state.name = self._joint_names
                # Return the response
                return PlanGraspResponse(True, stamped_ros_grasp_pose, ros_hand_joint_state)
        # In case of failure or shutdown return a response indicating failure.
        return PlanGraspResponse(False, PoseStamped(), JointState())


if __name__ == "__main__":
    rospy.init_node('hfts_planner_node')
    handler = HandlerClass()
    s = rospy.Service('/hfts_planner/plan_fingertip_grasp', PlanGrasp, handler.handle_plan_request)
    rospy.spin()

