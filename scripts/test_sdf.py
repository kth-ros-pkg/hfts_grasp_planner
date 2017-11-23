#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.robot as robot_sdf_module
import openravepy as orpy
import numpy as np

ENV_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/environments/table_r850.xml'
# ENV_PATH = '/home/joshua/projects/gpmp2_catkin/src/orgpmp2/examples/data/envs/lab.env.xml'
# ROBOT_PATH = '/home/joshua/projects/gpmp2_catkin/src/orgpmp2/examples/data/robots/barrettwam_gpmp2spheres.robot.xml'

if __name__=="__main__":
    env = orpy.Environment()
    env.Load(ENV_PATH)
    # env.Load(ROBOT_PATH)
    env.SetViewer('qtcoin')
    bodies_to_remove = ['fusebox', 'finder_dimmer', 'beckhoff']
    for body_name in bodies_to_remove:
        body = env.GetKinBody(body_name)
        env.Remove(body)
    movable_names = ['crayola', 'bunny']
    robot_name = 'r850_robotiq'
    scene_sdf = sdf_module.SceneSDF(env, movable_names, robot_name)
    volume = np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.2])
    # scene_sdf.create_sdf(volume, 0.02, 0.02)
    sdf_vis = sdf_module.ORSDFVisualization(env)
    robot_sdf = robot_sdf_module.RobotSDF(env.GetRobot(robot_name), scene_sdf)
    # sdf_builder = sdf.SDFBuilder(env, 0.2)
    # my_sdf = sdf_builder.create_sdf(np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.5]))
    # my_sdf.init_sdf(np.array([-1.2, -1.2, -1.2, 1.2, 1.2, 1.2]), 0.02)
    # my_sdf.init_sdf(np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.5]), 0.2)
    # my_sdf.visualize(0.8)

    scene_sdf.load('/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/sdfs/test.sdf')
    # sdf_vis.visualize(scene_sdf, volume, resolution=0.05, max_sat_value=0.7)
    robot_sdf.load_approximation('/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml')
    IPython.embed()