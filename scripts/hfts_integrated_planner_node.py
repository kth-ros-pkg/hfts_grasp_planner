#! /usr/bin/python

import IPython
import sys
import rospy
import rosgraph.roslogging
import rospkg
import logging
from hfts_grasp_planner.srv import PlanGraspMotion, PlanGraspMotionRequest, PlanGraspMotionResponse
from hfts_grasp_planner.integrated_hfts_planner import IntegratedHFTSPlanner


PACKAGE_NAME = 'hfts_grasp_planner'


class HandlerClass(object):
    """ Class that provides a ROS service callback for executing the
        integrated HFTS grasp planner."""

    def __init__(self):
        """ Creates a new handler class."""
        rospack = rospkg.RosPack()
        self._package_path = rospack.get_path(PACKAGE_NAME)
        # Create planner
        b_visualize_grasps = rospy.get_param('visualize_grasps', default=False)
        b_visualize_system = rospy.get_param('visualize_system', default=False)
        b_visualize_hfts = rospy.get_param('visualize_hfts', default=False)
        env_file = self._package_path + '/' + rospy.get_param('environment_file_name')
        p_goal_max = rospy.get_param('p_goal_max', default=0.8)
        p_goal_w = rospy.get_param('p_goal_w', default=1.2)
        p_goal_min = rospy.get_param('p_goal_min', default=0.01)
        min_iterations = rospy.get_param('min_iterations', default=20)
        max_iterations = rospy.get_param('max_iterations', default=70)
        hand_file = self._package_path + '/' + rospy.get_param('hand_file')
        robot_name = rospy.get_param('robot_name')
        manip_name = rospy.get_param('manipulator_name')
        # robot_file = self._package_path + rospy.get_param('robot_file')
        free_space_weight = rospy.get_param('free_space_weight', default=0.5)
        connected_space_weight = rospy.get_param('connected_space_weight', default=4.0)
        use_approximates = rospy.get_param('use_approximates', default=True)
        time_limit = rospy.get_param('time_limit', default=60.0)
        # Make sure we do not visualize grasps and the system at the same time (only one OR viewer)
        b_visualize_grasps = b_visualize_grasps and not b_visualize_system
        self._planner = IntegratedHFTSPlanner(env_file=env_file, robot_name=robot_name, manipulator_name=manip_name,
                                              b_visualize_system=b_visualize_system,
                                              b_visualize_grasps=b_visualize_grasps, b_visualize_hfts=b_visualize_hfts,
                                              hand_file=hand_file,
                                              min_iterations=min_iterations, max_iterations=max_iterations,
                                              free_space_weight=free_space_weight,
                                              connected_space_weight=connected_space_weight,
                                              use_approximates=use_approximates,
                                              time_limit=time_limit)

    # def handle_plan_request(self, req):
    #     """ Callback function for a grasp planning servce request. """
    #     # TODO generate HFTS from point cloud if point cloud is specified
    #     # pointCloud = req.point_cloud
    #     # Load the requested object first
    #     self._planner.loadObj(self._package_path + '/data', req.object_identifier)
    #     # We always start from the root node, so create a root node
    #     root_hfts_node = HFTSNode()
    #     max_iterations = rospy.get_param('max_iterations', 20)
    #     iteration = 0
    #     # Iterate until either shutdown, max_iterations reached or a good grasp was found
    #     while iteration < max_iterations and not rospy.is_shutdown():
    #         return_node = self._planner.sampleGrasp(root_hfts_node, 30)
    #         if return_node.isGoal():
    #             grasp_pose = return_node.getHandTransform()
    #             pose_quaternion = tff.quaternion_from_matrix(grasp_pose)
    #             pose_position = grasp_pose[:3, -1]
    #             # Save pose in ROS pose
    #             ros_grasp_pose = Pose()
    #             ros_grasp_pose.position.x = pose_position[0]
    #             ros_grasp_pose.position.y = pose_position[1]
    #             ros_grasp_pose.position.z = pose_position[2]
    #             ros_grasp_pose.orientation.x = pose_quaternion[0]
    #             ros_grasp_pose.orientation.y = pose_quaternion[1]
    #             ros_grasp_pose.orientation.z = pose_quaternion[2]
    #             ros_grasp_pose.orientation.w = pose_quaternion[3]
    #             # Make a header for the message
    #             header = Header()
    #             header.frame_id = req.object_identifier
    #             header.stamp = rospy.Time.now()
    #             # Create stamped pose
    #             stamped_ros_grasp_pose = PoseStamped()
    #             stamped_ros_grasp_pose.pose = ros_grasp_pose
    #             stamped_ros_grasp_pose.header = header
    #             # Create JointState message to send hand configuration
    #             hand_conf = return_node.getHandConfig()
    #             ros_hand_joint_state = JointState()
    #             ros_hand_joint_state.header = header
    #             ros_hand_joint_state.position = hand_conf
    #             ros_hand_joint_state.name = self._joint_names
    #             # Return the response
    #             return PlanGraspResponse(True, stamped_ros_grasp_pose, ros_hand_joint_state)
    #     # In case of failure or shutdown return a response indicating failure.
    #     return PlanGraspResponse(False, PoseStamped(), JointState())

    def handle_plan_request(self, request):
        # TODO wanna set the id as it is in the scene from somewhere else? I.e. not hardcoded
        self._planner.load_object(obj_file_path=self._package_path + '/data', obj_id=request.object_identifier,
                                  obj_id_scene='test_object')
        # TODO read start configuration from request and transform to OpenRAVE configuration
        start_config = 9 * [0.0]
        start_config[-1] = 1.0
        result = self._planner.plan(start_config)

if __name__ == "__main__":
    rospy.init_node('hfts_integrated_planner_node', log_level=rospy.DEBUG)
    # reconnect logging calls to ros log system
    logging.getLogger().addHandler(rosgraph.roslogging.RosStreamHandler())
    # logs sent to children of trigger with a level >= this will be redirected to ROS
    logging.getLogger().setLevel(logging.DEBUG)
    # Build handler
    handler = HandlerClass()
    s = rospy.Service('/hfts_planner/plan_fingertip_grasp_motion', PlanGraspMotion, handler.handle_plan_request)
    # rospy.spin()
    class DummyRequest:
        object_identifier = 'bunny'
    request = DummyRequest()
    handler.handle_plan_request(request)
    print 'Execution finished'
    sys.exit(0)