#! /usr/bin/python

import os
import argparse
import IPython
import numpy as np
import openravepy as orpy
from hfts_grasp_planner.utils import ObjectFileIO, OpenRAVEDrawer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object_name', help="Object to show HFTS for", type=str)
    parser.add_argument('--depth_level', help="HFTS level to show", type=int, default=1)
    parser.add_argument('--show_normals', help="If true, show normals", action="store_true")
    args = parser.parse_args()
    # do stuff
    env = orpy.Environment()
    env.SetViewer('qtcoin')
    base_path = os.path.dirname(__file__) + "/../"
    object_loader = ObjectFileIO(base_path + '/data/')
    hfts, hfts_params, com = object_loader.get_hfts(args.object_name)
    com_handle = env.drawbox(com, np.array([0.005, 0.005, 0.005]), np.array([1.0, 0.0, 0.0]))
    drawer = OpenRAVEDrawer(env, None, False)
    handles = object_loader.show_hfts(args.depth_level, drawer, b_normals=args.show_normals)
    print "Exit the program by typing \'quit\'"
    IPython.embed()
