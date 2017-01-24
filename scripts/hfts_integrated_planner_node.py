#! /usr/bin/python

import IPython
import sys
import rospy
import rosgraph.roslogging
import rospkg
import logging
import numpy
import xml.etree.ElementTree as xml_et
from hfts_grasp_planner.srv import PlanGraspMotion, PlanGraspMotionRequest, PlanGraspMotionResponse
from hfts_grasp_planner.cfg import IntegratedHFTSPlannerConfig
from hfts_grasp_planner.integrated_hfts_planner import IntegratedHFTSPlanner
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from dynamic_reconfigure.server import Server


PACKAGE_NAME = 'hfts_grasp_planner'


class HandlerClass(object):
    """ Class that provides a ROS service callback for executing the
        integrated HFTS grasp planner."""

    def __init__(self):
        """ Creates a new handler class."""
        # Get some system information
        rospack = rospkg.RosPack()
        self._package_path = rospack.get_path(PACKAGE_NAME)
        # Initialize member variables
        self._recent_joint_state = None
        self._params = {}
        # Update static parameters
        self._joint_state_topic_name = rospy.get_param('joint_state_topic', '/robot/joint_states')
        b_visualize_grasps = rospy.get_param('visualize_grasps', default=False)
        b_visualize_system = rospy.get_param('visualize_system', default=False)
        b_visualize_hfts = rospy.get_param('visualize_hfts', default=False)
        env_file = self._package_path + '/' + rospy.get_param('environment_file_name')
        hand_file = self._package_path + '/' + rospy.get_param('hand_file')
        robot_name = rospy.get_param('robot_name')
        manip_name = rospy.get_param('manipulator_name')
        # TODO remove this again and set these parameters only in the service callback
        # Load dynamic parameters also from the parameter server
        self._params['min_iterations'] = rospy.get_param('min_iterations', default=20)
        self._params['max_iterations'] = rospy.get_param('max_iterations', default=70)
        self._params['free_space_weight'] = rospy.get_param('free_space_weight', default=0.5)
        self._params['connected_space_weight'] = rospy.get_param('connected_space_weight', default=4.0)
        self._params['use_approximates'] = rospy.get_param('use_approximates', default=True)
        self._params['compute_velocities'] = rospy.get_param('compute_velocities', default=True)
        self._params['time_limit'] = rospy.get_param('time_limit', default=60.0)
        # Make sure we do not visualize grasps and the system at the same time (only one OR viewer)
        b_visualize_grasps = b_visualize_grasps and not b_visualize_system
        # Create planner
        self._planner = IntegratedHFTSPlanner(env_file=env_file, robot_name=robot_name, manipulator_name=manip_name,
                                              b_visualize_system=b_visualize_system,
                                              b_visualize_grasps=b_visualize_grasps, b_visualize_hfts=b_visualize_hfts,
                                              hand_file=hand_file,
                                              min_iterations=self._params['min_iterations'],
                                              max_iterations=self._params['max_iterations'],
                                              free_space_weight=self._params['free_space_weight'],
                                              connected_space_weight=self._params['connected_space_weight'],
                                              use_approximates=self._params['use_approximates'],
                                              compute_velocities=self._params['compute_velocities'],
                                              time_limit=self._params['time_limit'])
        # Listen to joint states
        rospy.Subscriber(self._joint_state_topic_name, JointState,
                         self.receive_joint_state)

    def update_parameters(self, config, level):
        # self._params['p_goal_max'] = rospy.get_param('p_goal_max', default=0.8)
        # self._params['p_goal_w'] = rospy.get_param('p_goal_w', default=1.2)
        # self._params['p_goal_min'] = rospy.get_param('p_goal_min', default=0.01)
        # TODO for some reason the default value for parameters of type double are ignored
        self._params = config
        return config

    def receive_joint_state(self, msg):
        self._recent_joint_state = msg

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

    def convert_configuration(self, configuration):
        output_config = None
        or_robot = self._planner.get_robot()
        if type(configuration) is JointState:
            # Convert from Joint state to OpenRAVE (list-type)
            name_position_mapping = {}
            output_config = numpy.zeros(or_robot.GetDOF())
            for i in range(len(configuration.name)):
                name_position_mapping[configuration.name[i]] = configuration.position[i]
            for joint in or_robot.GetJoints():
                output_config[joint.GetDOFIndex()] = name_position_mapping[joint.GetName()]
        elif type(configuration) is list or type(configuration) is numpy.array:
            # Convert from OpenRAVE (list-type) to Joint state
            output_config = JointState
            for i in range(len(configuration)):
                joint = or_robot.GetJointFromDOFIndex(i)
                output_config.name.append(joint.GetName())
                output_config.position.append(configuration[i])
        return output_config

    def convert_trajectory(self, traj):
        # The configuration specification allows us to interpret the trajectory data
        specs = traj.GetConfigurationSpecification()
        ros_trajectory = JointTrajectory()
        robot = self._planner.get_robot()
        manip = robot.GetActiveManipulator()
        indices = numpy.concatenate((manip.GetArmIndices(), manip.GetGripperIndices()))
        ros_trajectory.joint_names = map(lambda x: x.GetName(), robot.GetJoints())
        time_from_start = 0.0
        # Iterate over all waypoints
        for i in range(traj.GetNumWaypoints()):
            wp = traj.GetWaypoint(i)
            ros_traj_point = JointTrajectoryPoint()
            ros_traj_point.positions = specs.ExtractJointValues(wp, robot, range(robot.GetDOF()))
            ros_traj_point.velocities = specs.ExtractJointValues(wp, robot, range(robot.GetDOF()), 1)
            time_from_start += specs.ExtractDeltaTime(wp)
            ros_traj_point.time_from_start = rospy.Duration(time_from_start)
            ros_trajectory.points.append(ros_traj_point)
        return ros_trajectory

    def handle_plan_request(self, request):
        # TODO: update planner parameters
        rospy.loginfo('Executing planner with parameters: ' + str(self._params))
        self._planner.set_parameters(time_limit=self._params['time_limit'],
                                     compute_velocities=self._params['compute_velocities'])
        response = PlanGraspMotionResponse()
        obj_id = request.object_identifier
        model_id = request.model_identifier
        if len(model_id) == 0:
            model_id = None
        # Prepare the planner to work with the target object
        self._planner.load_object(obj_file_path=self._package_path + '/data', obj_id=obj_id,
                                  model_id=model_id)
        # Get the start configuration
        ros_start_configuration = request.start_configuration
        if len(ros_start_configuration.position) == 0:
            # no start configuration given, get it from robot state
            ros_start_configuration = self._recent_joint_state
        if ros_start_configuration is None or len(ros_start_configuration.position) == 0:
            rospy.logerr('Start configuration unknown. There is neither a configuration given in the service request ' +
                         'nor was any joint state received from ' + self._joint_state_topic_name)
            response.planning_success = False
            return response
        start_config = self.convert_configuration(ros_start_configuration)
        # PLAN
        result = self._planner.plan(start_config)
        # Process result
        if result is None:
            response.planning_success = False
            return response
        ros_trajectory = self.convert_trajectory(result)
        response.trajectory = ros_trajectory
        response.planning_success = True
        return response

if __name__ == "__main__":
    rospy.init_node('hfts_integrated_planner_node', log_level=rospy.DEBUG)
    # reconnect logging calls to ros log system
    logging.getLogger().addHandler(rosgraph.roslogging.RosStreamHandler())
    # logs sent to children of trigger with a level >= this will be redirected to ROS
    logging.getLogger().setLevel(logging.DEBUG)
    # Build handler
    handler = HandlerClass()
    srv = Server(IntegratedHFTSPlannerConfig, handler.update_parameters)
    s = rospy.Service('/hfts_planner/plan_fingertip_grasp_motion', PlanGraspMotion, handler.handle_plan_request)
    rospy.spin()
    # class DummyRequest:
    #     object_identifier = 'crayola'
    # request = DummyRequest()
    # handler.handle_plan_request(request)
    # print 'Execution finished'
    sys.exit(0)