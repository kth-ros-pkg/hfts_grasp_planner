#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf
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
    my_sdf = sdf.SDF(env=env)
    # sdf.init_sdf(np.array([-1.2, -1.2, -1.2, 1.2, 1.2, 1.2]), 0.02)
    # my_sdf.init_sdf(np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.5]), 0.2)
    # sdf.visualize(0.8)
    IPython.embed()