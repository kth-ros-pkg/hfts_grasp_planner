#! /usr/bin/python

import rospy
import rosgraph.roslogging
import rospkg
import logging
import numpy
import tf.transformations
from hfts_grasp_planner.srv import *
from hfts_grasp_planner.cfg import IntegratedHFTSPlannerConfig
from hfts_grasp_planner.integrated_hfts_planner import IntegratedHFTSPlanner
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from dynamic_reconfigure.server import Server


PACKAGE_NAME = 'hfts_grasp_planner'


class HandlerClass(object):
    """ Class that provides a ROS service callback for executing the
        integrated HFTS grasp planner."""

    JOINT_VALUE_NOISE = 0.001

    def __init__(self):
        """ Creates a new handler class."""
        # Get some system information
        rospack = rospkg.RosPack()
        self._package_path = rospack.get_path(PACKAGE_NAME)
        # Initialize member variables
        self._recent_joint_state = None
        self._params = {}
        # Update static parameters
        node_name = rospy.get_name()
        self._joint_state_topic_name = rospy.get_param(node_name + '/joint_state_topic',
                                                       default='/robot/joint_states')
        self._joint_names_mapping = rospy.get_param(node_name + '/joint_names_mapping', default=None)
        self._inverted_joint_names_mapping = self._invert_joint_names_mapping(self._joint_names_mapping)
        assert (self._joint_names_mapping is None and self._inverted_joint_names_mapping is None) \
            or (self._joint_names_mapping is not None and self._inverted_joint_names_mapping is not None)
        b_visualize_grasps = rospy.get_param(node_name + '/visualize_grasps', default=False)
        b_visualize_system = rospy.get_param(node_name + '/visualize_system', default=False)
        b_visualize_hfts = rospy.get_param(node_name + '/visualize_hfts', default=False)
        b_show_traj = rospy.get_param(node_name + '/show_trajectory', default=False)
        b_show_search_tree = rospy.get_param(node_name + '/show_search_tree', default=False)
        env_file = self._package_path + '/' + rospy.get_param(node_name + '/environment_file_name')
        hand_file = self._package_path + '/' + rospy.get_param(node_name + '/hand_file')
        hand_cache_file = self._package_path + '/' + rospy.get_param(node_name + '/hand_cache_file')
        robot_name = rospy.get_param(node_name + '/robot_name')
        manip_name = rospy.get_param(node_name + '/manipulator_name')
        # Load dynamic parameters also from the parameter server
        # Make sure we do not visualize grasps and the system at the same time (only one OR viewer)
        b_visualize_grasps = b_visualize_grasps and not b_visualize_system
        # Create planner
        self._planner = IntegratedHFTSPlanner(env_file=env_file,
                                              hand_cache_file=hand_cache_file,
                                              robot_name=robot_name,
                                              manipulator_name=manip_name,
                                              data_root_path=self._package_path + '/data',
                                              b_visualize_system=b_visualize_system,
                                              b_visualize_grasps=b_visualize_grasps,
                                              b_visualize_hfts=b_visualize_hfts,
                                              b_show_traj=b_show_traj,
                                              b_show_search_tree=b_show_search_tree,
                                              hand_file=hand_file)
        # Listen to joint states
        rospy.Subscriber(self._joint_state_topic_name, JointState,
                         self.receive_joint_state)

    def _invert_joint_names_mapping(self, input_mapping):
        if input_mapping is None:
            return None
        inverted_mapping = {}
        for key, value in self._joint_names_mapping.iteritems():
            inverted_mapping[value] = key
        return inverted_mapping

    def update_parameters(self, config, level):
        # self._params['p_goal_max'] = rospy.get_param('p_goal_max', default=0.8)
        # self._params['p_goal_w'] = rospy.get_param('p_goal_w', default=1.2)
        # self._params['p_goal_min'] = rospy.get_param('p_goal_min', default=0.01)
        # TODO for some reason the default value for parameters of type double are ignored
        self._params = config
        return config

    def receive_joint_state(self, msg):
        self._recent_joint_state = msg

    def clamp_joint_value(self, value, joint):
        min_value, max_value = joint.GetLimits()
        if value < min_value - self.JOINT_VALUE_NOISE:
            rospy.logerr('The joint value of joint %s is too small.' % joint.GetName())
        elif value > max_value + self.JOINT_VALUE_NOISE:
            rospy.logerr('The joint value of joint %s is too large.' % joint.GetName())
        return min(max(value, min_value + self.JOINT_VALUE_NOISE), max_value - self.JOINT_VALUE_NOISE)

    def convert_pose(self, ros_pose=None, numpy_pose=None):
        if ros_pose is not None:
            tf_matrix = numpy.eye(4)
            if type(ros_pose) is PoseStamped:
                # TODO we need to transform the pose first using tf
                if ros_pose.header.frame_id != 'world':
                    tf_matrix = self._planner.get_object_frame(ros_pose.header.frame_id)
                    if tf_matrix is None:
                        rospy.logerr('Pose frame %s unknown.' % ros_pose.header.frame_id)
                        return None
                ros_pose = ros_pose.pose

            transform_matrix = tf.transformations.quaternion_matrix([ros_pose.orientation.x,
                                                                     ros_pose.orientation.y,
                                                                     ros_pose.orientation.z,
                                                                     ros_pose.orientation.w])
            transform_matrix[:3, 3] = [ros_pose.position.x,
                                       ros_pose.position.y,
                                       ros_pose.position.z]
            return numpy.dot(tf_matrix, transform_matrix)
        if numpy_pose is not None:
            ros_pose = Pose()
            ros_pose.position.x = numpy_pose[0, 3]
            ros_pose.position.y = numpy_pose[1, 3]
            ros_pose.position.z = numpy_pose[2, 3]
            quaternion = tf.transformations.quaternion_from_matrix(numpy_pose)
            ros_pose.orientation.x = quaternion[0]
            ros_pose.orientation.y = quaternion[1]
            ros_pose.orientation.z = quaternion[2]
            ros_pose.orientation.w = quaternion[3]
            return ros_pose
        return None

    def convert_configuration(self, configuration):
        output_config = None
        or_robot = self._planner.get_robot()
        if type(configuration) is JointState:
            configuration.name = self._fix_joint_names(configuration.name, 'openrave')
            # Convert from Joint state to OpenRAVE (list-type)
            name_position_mapping = {}
            output_config = numpy.zeros(or_robot.GetDOF())
            for i in range(len(configuration.name)):
                name_position_mapping[configuration.name[i]] = configuration.position[i]
            for joint in or_robot.GetJoints():
                output_config[joint.GetDOFIndex()] = self.clamp_joint_value(name_position_mapping[joint.GetName()], joint)
        elif type(configuration) is list or type(configuration) is numpy.array:
            # Convert from OpenRAVE (list-type) to Joint state
            output_config = JointState()
            for i in range(len(configuration)):
                joint = or_robot.GetJointFromDOFIndex(i)
                output_config.name.append(joint.GetName())
                output_config.position.append(configuration[i])
                output_config.name = self._fix_joint_names(output_config.name, 'ros')
        return output_config

    def convert_trajectory(self, traj, arm_only=False):
        # The configuration specification allows us to interpret the trajectory data
        specs = traj.GetConfigurationSpecification()
        ros_trajectory = JointTrajectory()
        robot = self._planner.get_robot()
        manip = robot.GetActiveManipulator()
        if arm_only:
            dof_indices = manip.GetArmIndices()
        else:
            dof_indices = numpy.concatenate((manip.GetArmIndices(), manip.GetGripperIndices()))
        joint_names = map(lambda x: x.GetName(), robot.GetJoints())
        ros_trajectory.joint_names = [joint_names[i] for i in dof_indices]
        ros_trajectory.joint_names = self._fix_joint_names(ros_trajectory.joint_names, 'ros')
        time_from_start = 0.0
        # Iterate over all waypoints
        for i in range(traj.GetNumWaypoints()):
            wp = traj.GetWaypoint(i)
            ros_traj_point = JointTrajectoryPoint()
            ros_traj_point.positions = specs.ExtractJointValues(wp, robot, range(len(dof_indices)))
            ros_traj_point.velocities = specs.ExtractJointValues(wp, robot, range(len(dof_indices)), 1)
            delta_t = specs.ExtractDeltaTime(wp)
            # TODO why does this happen?
            if delta_t <= 10e-8 and i > 0:
                rospy.logwarn('We have redundant waypoints in this trajectory, skipping...')
                continue
            time_from_start += delta_t
            rospy.loginfo('Delta t is : %f' % delta_t)
            ros_traj_point.time_from_start = rospy.Duration().from_sec(time_from_start)
            ros_trajectory.points.append(ros_traj_point)

        return ros_trajectory

    def _fix_joint_names(self, joint_names, target_name):
        # We might have different joint names in the collada model and the ros environment
        # we are working in. In this case the user can specify a mapping on the parameter server
        # If such a mapping is defined, self._joint_names_mapping maps openrave names to ros names,
        # and self._inverted_joint_names_mapping vice versa.
        if self._joint_names_mapping is None:
            return joint_names
        if target_name == 'ros':
            for name_idx in range(len(joint_names)):
                or_name = joint_names[name_idx]
                if or_name in self._joint_names_mapping:
                    joint_names[name_idx] = self._joint_names_mapping[or_name]
        elif target_name == 'openrave':
            for name_idx in range(len(joint_names)):
                ros_name = joint_names[name_idx]
                if ros_name in self._inverted_joint_names_mapping:
                    joint_names[name_idx] = self._inverted_joint_names_mapping[ros_name]
        else:
            raise ValueError('Unknown target %s for joint name mapping' % str(target_name))
        return joint_names

    def get_start_configuration(self, ros_start_configuration):
        # Get the start configuration
        if len(ros_start_configuration.position) == 0:
            # no start configuration given, get it from robot state
            ros_start_configuration = self._recent_joint_state
        if ros_start_configuration is None or len(ros_start_configuration.position) == 0:
            rospy.logerr('Start configuration unknown. There is neither a configuration given in the service request ' +
                         'nor was any joint state received from ' + self._joint_state_topic_name)
            return None
        return self.convert_configuration(ros_start_configuration)

    def handle_plan_request(self, request):
        # TODO: update planner parameters
        rospy.loginfo('Executing planner with parameters: ' + str(self._params))
        hfts_gen_params = {'max_normal_variance': self._params['max_normal_variance'],
                           'min_contact_patch_radius': self._params['min_contact_patch_radius'],
                           'contact_density': self._params['contact_density'],
                           'max_num_points': self._params['max_num_points'],
                           'position_weight': self._params['hfts_position_weight'],
                           'branching_factor': self._params['hfts_branching_factor'],
                           'first_level_branching_factor': self._params['hfts_first_level_branching_factor']}
        self._planner.set_parameters(time_limit=self._params['time_limit'],
                                     compute_velocities=self._params['compute_velocities'],
                                     com_center_weight=self._params['com_center_weight'],
                                     hfts_generation_params=hfts_gen_params,
                                     b_force_new_hfts=self._params['force_new_hfts'],
                                     min_iterations=self._params['min_iterations'],
                                     max_iterations=self._params['max_iterations'],
                                     free_space_weight=self._params['free_space_weight'],
                                     connected_space_weight=self._params['connected_space_weight'],
                                     max_num_hierarchy_descends=self._params['max_num_hierarchy_descends'],
                                     use_approximates=self._params['use_approximates'],
                                     vel_factor=self._params['velocity_factor'])
        response = PlanGraspMotionResponse()
        obj_id = request.object_identifier
        model_id = request.model_identifier
        if len(model_id) == 0:
            model_id = None
        # Prepare the planner to work with the target object
        self._planner.load_target_object(obj_id=obj_id,
                                         model_id=model_id)
        # Get start config
        start_config = self.get_start_configuration(request.start_configuration)
        if start_config is None:
            response.planning_success = False
            return response
        rospy.loginfo('Planning from configuration ' + str(start_config))
        # PLAN
        result, grasp_pose = self._planner.plan(start_config)
        # Process result
        if result is None:
            response.planning_success = False
            return response
        ros_trajectory = self.convert_trajectory(result)
        response.trajectory = ros_trajectory
        response.planning_success = True
        response.grasp_pose.pose = self.convert_pose(numpy_pose=grasp_pose)
        response.grasp_pose.header = Header(stamp=rospy.Time.now(),
                                            frame_id=request.object_identifier)
        return response

    def handle_move_arm_request(self, request):
        response = PlanArmMotionResponse(planning_success=False)
        target_pose = self.convert_pose(ros_pose=request.target_pose)
        if target_pose is None:
            return response
        start_configuration = self.get_start_configuration(request.start_configuration)
        if start_configuration is None:
            response.planning_success = False
            return response
        grasped_object = request.grasped_object
        if len(grasped_object) == 0:
            grasped_object = None
        traj = self._planner.plan_arm_motion(target_pose, start_configuration, grasped_object=grasped_object)
        if traj is not None:
            response.trajectory = self.convert_trajectory(traj, arm_only=True)
            response.planning_success = True
        return response

    def handle_add_object_request(self, request):
        response = AddObjectResponse(success=False)
        # TODO we need to transform this pose to world frame using TF
        transform_matrix = self.convert_pose(ros_pose=request.pose)
        response.success = self._planner.add_planning_scene_object(object_name=request.object_identifier,
                                                                   object_class_name=request.class_identifier,
                                                                   pose=transform_matrix)
        return response

    def handle_remove_object_request(self, request):
        response = RemoveObjectResponse()
        response.success = self._planner.remove_planning_scene_object(object_name=request.object_identifier)
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
    # Create services
    add_obj_service = rospy.Service(rospy.get_name() + '/add_object',
                                    AddObject, handler.handle_add_object_request)
    remove_obj_service = rospy.Service(rospy.get_name() + '/remove_object',
                                       RemoveObject, handler.handle_remove_object_request)
    planning_service = rospy.Service(rospy.get_name() + '/plan_fingertip_grasp_motion',
                                     PlanGraspMotion, handler.handle_plan_request)
    arm_planning_service = rospy.Service(rospy.get_name() + '/plan_arm_motion',
                                         PlanArmMotion, handler.handle_move_arm_request)
    # Spin until node is killed
    rospy.spin()
    sys.exit(0)