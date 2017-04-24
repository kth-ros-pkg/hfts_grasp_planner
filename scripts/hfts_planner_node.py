#! /usr/bin/python

import rospy
import rospkg
from hfts_grasp_planner.utils import *
from hfts_grasp_planner.core import HFTSSampler, HFTSNode
from hfts_grasp_planner.srv import PlanGrasp, PlanGraspRequest, PlanGraspResponse
from hfts_grasp_planner.cfg import HFTSPlannerConfig
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
import tf.transformations as tff
from std_msgs.msg import Header
from dynamic_reconfigure.server import Server


PACKAGE_NAME = 'hfts_grasp_planner'


class HandlerClass(object):
    """ Class that provides a ROS service callback for executing the HFTS grasp planner."""

    def __init__(self):
        """ Creates a new handler class."""
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(PACKAGE_NAME)
        self._params = {}
        # Update static parameters
        b_visualize = rospy.get_param(rospy.get_name() + '/visualize', default=False)
        # Update dynamic parameters
        # Create planner
        self._object_loader = ObjectFileIO(package_path + '/data/')
        self._planner = HFTSSampler(self._object_loader, num_hops=4, vis=b_visualize)
        # Load hand and save joint names
        hand_file = package_path + rospy.get_param(rospy.get_name() + '/hand_file')
        hand_cache_file = package_path + '/' + rospy.get_param(rospy.get_name() + '/hand_cache_file')
        self._planner.load_hand(hand_file, hand_cache_file)
        or_hand = self._planner.get_or_hand()
        joints = or_hand.GetJoints()
        self._joint_names = []
        for joint in joints:
            self._joint_names.append(joint.GetName())

    def handle_plan_request(self, req):
        """ Callback function for a grasp planning servce request. """
        # TODO generate HFTS from point cloud if point cloud is specified
        # pointCloud = req.point_cloud
        rospy.loginfo('Executing planner with parameters: ' + str(self._params))
        # Load the requested object first
        # TODO setting this boolean parameter should be solved in a more elegant manner
        self._object_loader._b_var_filter = self._params['hfts_filter_points']
        self._planner.load_object(req.object_identifier)
        hfts_gen_params = {'max_normal_variance': self._params['max_normal_variance'],
                           'contact_density': self._params['contact_density'],
                           'min_contact_patch_radius': self._params['min_contact_patch_radius'],
                           'max_num_points': self._params['max_num_points'],
                           'position_weight': self._params['hfts_position_weight'],
                           'branching_factor': self._params['hfts_branching_factor'],
                           'first_level_branching_factor': self._params['hfts_first_level_branching_factor']}
        self._planner.set_parameters(max_iters=self._params['num_hfts_iterations'],
                                     reachability_weight=self._params['reachability_weight'],
                                     com_center_weight=self._params['com_center_weight'],
                                     hfts_generation_params=hfts_gen_params,
                                     b_force_new_hfts=self._params['force_new_hfts'])
        # We always start from the root node, so create a root node
        root_hfts_node = HFTSNode()
        num_planning_attempts = self._params['num_planning_attempts']
        rospy.loginfo('[HandlerClass::handle_plan_request] Planning grasp, running %i attempts.' % num_planning_attempts)
        iteration = 0
        # Iterate until either shutdown, max_iterations reached or a good grasp was found
        while iteration < num_planning_attempts and not rospy.is_shutdown():
            return_node = self._planner.sample_grasp(root_hfts_node,
                                                     self._planner.get_maximum_depth(),
                                                     post_opt=True)
            iteration += 1
            if return_node.is_goal():
                rospy.loginfo('[HandlerClass::handle_plan_request] Found a grasp after %i attempts.' % iteration)
                grasp_pose = return_node.get_hand_transform()
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
                hand_conf = return_node.get_hand_config()
                ros_hand_joint_state = JointState()
                ros_hand_joint_state.header = header
                ros_hand_joint_state.position = hand_conf
                ros_hand_joint_state.name = self._joint_names
                # Return the response
                return PlanGraspResponse(True, stamped_ros_grasp_pose, ros_hand_joint_state)
        # In case of failure or shutdown return a response indicating failure.
        rospy.loginfo('[HandlerClass::handle_plan_request] Failed to find a grasp.')
        return PlanGraspResponse(False, PoseStamped(), JointState())

    def update_parameters(self, config, level):
        rospy.loginfo('[HandlerClass::update_parameters] Received new parameters.')
        self._params = config
        return config


if __name__ == "__main__":
    rospy.init_node('hfts_planner_node')
    handler = HandlerClass()
    srv = Server(HFTSPlannerConfig, handler.update_parameters)
    s = rospy.Service('/hfts_planner/plan_fingertip_grasp', PlanGrasp, handler.handle_plan_request)
    rospy.spin()

