# hfts_grasp_planner
A ROS-integrated implementation of the HFTS fingertip grasp planner.

## Installation
In order to run the hfts_planner node you need:

- ROS Indigo 
- OpenRAVE 0.8.2
- Python-sklearn
- Python-numpy
- Python-yaml

After intalling the dependencies above, install the package as follows:
```shell
cd $YOUR_CATKIN_WS/src
git clone https://github.com/kth-ros-pkg/hfts_grasp_planner
cd .. && catkin_make
```
## How to run
There is a launch file 'start_hfts_planner.launch', which starts the planner node and loads default 
parameters to the ROS parameter server. The default parameters can be modified in 
```shell
rosed $YOUR_CATKIN_WS/src/hfts_grasp_planner/config/testParams.yaml
```
Once the node is running, you can call the service manually:
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
