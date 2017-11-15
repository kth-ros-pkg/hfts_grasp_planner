#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf
import openravepy as orpy
import numpy as np

ENV_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/environments/table_r850.xml'

if __name__=="__main__":
    sdf = sdf.SDF(ENV_PATH)
    sdf._env.SetViewer('qtcoin')
    sdf.init_sdf(np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.5]), 0.2)
    sdf.visualize(0.8)
    IPython.embed()