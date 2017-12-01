#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.robot as robot_sdf_module
import hfts_grasp_planner.sdf.costs as costs_module
import openravepy as orpy
import numpy as np

ENV_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/environments/table_r850.xml'
SDF_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/sdfs/table_r850.sdf'
ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml'
# ENV_PATH = '/home/joshua/projects/gpmp2_catkin/src/orgpmp2/examples/data/envs/lab.env.xml'
# ROBOT_PATH = '/home/joshua/projects/gpmp2_catkin/src/orgpmp2/examples/data/robots/barrettwam_gpmp2spheres.robot.xml'

# def test_ball_sdf(volume, radius, resolution):
#     env = orpy.Environment()
#     ball_body = orpy.RaveCreateKinBody(env, '')
#     ball_body.SetName('ball')
#     ball_body.InitFromSpheres(np.array([[0.0, 0.0, 0.0, radius]]))
#     env.AddKinBody(ball_body)
#     scene_sdf = sdf_module.SceneSDF(env, [])
#     scene_sdf.create_sdf(volume, resolution)
#     num_samples = (volume[3:] - volume[:3]) / resolution
#     positions = np.array([np.array([x, y, z, 1.0]) for x in np.linspace(volume[0], volume[3] - resolution, num_samples[0])
#                            for y in np.linspace(volume[1], volume[4] - resolution, num_samples[1])
#                            for z in np.linspace(volume[2], volume[5] - resolution, num_samples[2])])
#     values = scene_sdf.get_distances(positions)
#     errors = np.zeros((positions.shape[0], 2))
#     for idx in xrange(positions.shape[0]):
#         true_distance = np.linalg.norm(positions[idx, :3]) - radius
#         errors[idx, 0] = true_distance - values[idx]
#         if abs(true_distance) > 0.0001:
#             errors[idx, 1] = errors[idx, 0] / true_distance
#     print('Mean error is ' + str(np.mean(errors[:, 0])))
#     print('Std is ' + str(np.std(errors[:, 0])))
#     import IPython
#     IPython.embed()


if __name__=="__main__":
    volume = np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.2])
    # test_ball_sdf(volume, 0.4, 0.01)
    env = orpy.Environment()
    env.Load(ENV_PATH)
    # env.Load(ROBOT_PATH)
    env.SetViewer('qtcoin')
    # bodies_to_remove = ['fusebox', 'finder_dimmer', 'beckhoff']
    # for body_name in bodies_to_remove:
    #     body = env.GetKinBody(body_name)
    #     env.Remove(body)
    movable_names = ['crayola', 'bunny']
    robot_name = 'r850_robotiq'
    robot = env.GetRobot(robot_name)
    scene_sdf = sdf_module.SceneSDF(env, movable_names, excluded_bodies=[robot_name])
    volume = np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.2])
    scene_sdf.load(SDF_PATH)
    # scene_sdf.create_sdf(volume, 0.02, 0.01)
    sdf_vis = sdf_module.ORSDFVisualization(env)
    # scene_sdf.load('/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/sdfs/test.sdf')
    # sdf_vis.visualize(scene_sdf, volume, resolution=0.05, max_sat_value=0.7)
    robot_sdf = robot_sdf_module.RobotSDF(robot, scene_sdf)
    robot_sdf.load_approximation(ROBOT_BALL_PATH)
    distance_fn = costs_module.DistanceToFreeSpace(robot, robot_sdf)
    IPython.embed()