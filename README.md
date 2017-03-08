# hfts_grasp_planner
A ROS-integrated implementation of the HFTS fingertip grasp planner.

## Installation
In order to run all the ROS nodes in this repository you need:

- ROS Indigo 
- OpenRAVE 0.8.2 (with Python bindings)
- scikit-learn (0.14.1) (Python)
- numpy (1.8.2) (Python)
- numpy-stl (2.2.0) (Python)
- PyYaml (Python)
- tf (Python)
- scipy (Python)
- python-igraph (0.6.5) (Python)
- matplotlib (Python)
- libspatialindex-dev (C library)
- Rtree (Python)

After intalling the dependencies above, install the package as follows:
```shell
cd $YOUR_CATKIN_WS/src
git clone https://github.com/kth-ros-pkg/hfts_grasp_planner
cd .. && catkin_make
```
## How to run
This repository contains two relevant ROS nodes: *hfts_integrated_planner_node* and *hfts_planner_node*.
The *hfts_integrated_planner_node* provides an integrated grasp and motion planning algorithm
whereas the *hfts_planner_node* provides only a grasp planning algorithm.
For both nodes there are launch files available.

#### How to use and launch hfts_planner_node
The launch file 'start_hfts_planner.launch' starts the *hfts_planner_node* and loads default 
parameters to the ROS parameter server. The default parameters can be modified in 
```shell
rosed $YOUR_CATKIN_WS/src/hfts_grasp_planner/config/testParams.yaml
```
Once this node is running, you can issue planning by calling the service */hfts_planner/plan_fingertip_grasp*.
You may call the service manually:
```shell
rosservice call /hfts_planner/plan_fingertip_grasp "object_identifier: 'bunny'
point_cloud:
  header:
    seq: 0
    stamp:
      secs: 0
      nsecs: 0
    frame_id: ''
  points:
  - x: 0.0
    y: 0.0
    z: 0.0
  channels:
  - name: ''
    values:
    - 0"
```
As a response you should receive a message containing a
hand pose relative to the object frame and a hand configuration.
See the service definition in *hfts_grasp_planner/srv/PlanGrasp.srv* for more details.

#### How to use and launch hfts_integrated_planner_node
The launch file 'start_integrated_hfts_planner.launch' launches
the *hfts_integrated_planner_node* and loads default 
parameters to the ROS parameter server. 
The default parameters can be modified in 
```shell
rosed $YOUR_CATKIN_WS/src/hfts_grasp_planner/config/integrated_hfts_params.yaml
```
**Note, that for this node to work you need to provide a robot model. 
See below for details.**

Once this node is running, you can issue planning by calling the service 
*/hfts_integrated_planner_node/plan_fingertip_grasp_motion*.
You may call the service manually:
```shell
rosservice call /hfts_integrated_planner_node/plan_fingertip_grasp_motion "object_identifier: 'bunny'
model_identifier: 'bunny'
point_cloud:
  header:
    seq: 0
    stamp: {secs: 0, nsecs: 0}
    frame_id: ''
  points:
  - {x: 0.0, y: 0.0, z: 0.0}
  channels:
  - name: ''
    values: [0]
start_configuration:
  header:
    seq: 0
    stamp: {secs: 0, nsecs: 0}
    frame_id: ''
  name: ['iiwa.joint0','iiwa.joint1', 'iiwa.joint2', 'iiwa.joint3', 'iiwa.joint4', 'iiwa.joint5', 'iiwa.joint6', 'scissor_joint', 'finger_2_joint_1']
  position: [0, 0, 0, 0, 0, 0, 0, 0.0, 0.5]
  velocity: [0]
  effort: [0]
object_pose:
  header:
    seq: 0
    stamp: {secs: 0, nsecs: 0}
    frame_id: ''
  pose:
    position: {x: 0.8, y: 0.8, z: 0.8}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}"
```
If planning a grasp was succesful, you should receive a message containing
a hand-arm path/trajectory to a grasping configuration and some additional information.
See the service definition in *hfts_grasp_planner/srv/PlanGraspMotion.srv*
for more details.

The planning scene can be modified using the services */hfts_integrated_grasp_planner_node/add_object* and
*/hfts_integrated_grasp_planner_node/remove_object*. Again, please consult the service definitions in *hfts_grasp_planner/srv/AddObject.srv*
and *hfts_grasp_planner/srv/RemoveObject.srv* for more details.

##Configuration 
Both nodes can be configured in different ways. Parameters that are required on startup or
not likely to change while a node is running are set on the ROS parameter server.
Parameters that directly affect the performance of the algorithms that a user
may want to change without restarting the node can be set using [*dynamic reconfigure*](http://wiki.ros.org/dynamic_reconfigure).

####Configuring hfts_planner_node
The parameters read by the *hfts_planner_node* from the parameter server are the following:
```yamlex
visualize: True
hand_cache_file: data/cache/robotiq_hand.npy
hand_file: models/robotiq/urdf_openrave_conversion/robotiq_s_thin.xml
```
The parameter *visualize* can either be *True* or *False* and determines whether to show
a window showing the robotic hand and the target object. If a grasp has been computed, the grasp is shown in the viewer.

The parameters *hand_cache_file* and *hand_file* are tightly coupled. Both parameters are strings containing the path 
and filenames. These paths need to be relative to the package. In case of *hand_file* the referred file should contain an OpenRAVE model of the robotic hand used for 
grasp planning. In case of *hand_cache_file* the file is used by the node to store hand-specific data in. 
In other words, you should change this parameter to a new filename whenever you select a new *hand_file*.

##### Dynamic configuration
There are several additional parameters that can be dynamically reconfigured. See
*hfts_grasp_planner/cfg/hfts_planner.cfg* for details.


####Configuring hfts_integrated_planner_node
The parameters read by the *hfts_integrated_planner_node* from the parameter server are the following:
```yamlex
visualize_grasps: False
visualize_system: True
visualize_hfts: False
show_trajectory: False
show_search_tree: False
hand_file: models/robotiq/urdf_openrave_conversion/robotiq_s_thin.xml
hand_cache_file: data/cache/robotiq_hand.npy
environment_file_name: data/environments/test_env.xml
robot_name: kmr_iiwa_robotiq
manipulator_name: arm_with_robotiq
joint_state_topic: /kmr/hand_arm_joint_states
joint_names_mapping:
  iiwa.joint0: lbrAxis1
  iiwa.joint1: lbrAxis2
  iiwa.joint2: lbrAxis3
  iiwa.joint3: lbrAxis4
  iiwa.joint4: lbrAxis5
  iiwa.joint5: lbrAxis6
  iiwa.joint6: lbrAxis7
```
The parameters *visualize_grasps* and *visualize_system* 
can either be *True* or *False* and determine whether to show
a window showing the whole environment, i.e. the full robot with its surrounding or
just a free-floating robotic hand with the target object. Note that at most one flag can be *True* at a time.
You can not display both at the same time.

The parameters *visualize_hfts* and *show_search_tree* are for debugging purposes.
In case of *visualize_hfts* the explored grasp search space is published to a ROS topic
*/hfts_integrated_planner_node/goal_region_graph*. This search space can be visualized using the node 
*hierarchy_visualizer_node.py*. If the parameter *show_search_tree* and *visualize_system* are *True*, a projection of the BiRRT is shown 
in the system viewer.

In case the parameter *show_trajectory* is *True*, every trajectory found by the planner will be executed on the simulated 
robot in OpenRAVE, allowing the user to see the trajectory.

The parameters *hand_cache_file* and *hand_file* are identical to the ones described for *hfts_planner_node*.

The parameter *environment_file_name* must be a path relative to the package to a file containing an OpenRAVE environment.
This environment needs to contain a robot with the same robotic hand as specified in *hand_file*.
The name of the robot in this environment must have the name *robot_name*. 

The parameter *manipulator_name* specifies which manipulator of the robot to use for planning.

Finally, the parameters *joint_state_topic* and *joint_names_mapping* serve for synchronization of the robot state.
The parameter *joint_state_topic* must be the name of a ROS topic on which the current joint states of the robot are published.
It is assumed that the published joint state contains the joint states for the whole robot, i.e. the arm and the hand.
The parameter *joint_names_mapping* is optional and may define a mapping from joint names used in the joint states messages
published on *joint_state_topic* to the joint names used in the OpenRAVE model.
If no joint states are published, a planning request always needs to specify the start configuration.

##### Dynamic configuration
There are several parameters additional that can be dynamically reconfigured. See
*hfts_grasp_planner/cfg/hfts_planner.cfg* for details.
