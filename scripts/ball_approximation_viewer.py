#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.robot as robot_sdf_module
import openravepy as orpy
import numpy as np

ROBOT_FILE = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/schunk-sdh/Schunk-SDH.robot.xml'
ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/schunk-sdh/ball_description.yaml'

if __name__=="__main__":
    env = orpy.Environment()
    env.Load(ROBOT_FILE)
    robot = env.GetRobots()[0]
    env.SetViewer('qtcoin')
    robot_sdf = robot_sdf_module.RobotSDF(robot)
    robot_sdf.load_approximation(ROBOT_BALL_PATH)
    IPython.embed()