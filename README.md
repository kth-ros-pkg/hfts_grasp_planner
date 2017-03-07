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
This repository contains two relevant ROS nodes: hfts_integrated_planner_node and hfts_planner_node.
The hfts_integrated_planner_node provides an integrated grasp and motion planning algorithm
whereas the hfts_planner_node provides only a grasp planning algorithm.
For both nodes there are launch files available.

#### How to use launch hfts_planner
The launch file 'start_hfts_planner.launch' starts the hfts_planner node and loads default 
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

#### How to use launch hfts_integrated_planner_node
The launch file 'start_integrated_hfts_planner.launch' launches
the hfts_integrated_planner_node and loads default 
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

##Configuration 
**TODO**
