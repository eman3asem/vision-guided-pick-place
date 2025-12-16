import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import random
import mujoco

from ompl import base as ob
from ompl import geometric as og

from robot import *
from exercises.cv_demo import program
from cam import get_pointcloud, show_pointcloud, get_camera_pose_cv
from exercises.Project import program, via_points

def program(d, m):
    # Define our robot object
    robot = UR5robot(data=d, model=m)

    pose = get_camera_pose_cv(d, m)
    pointcloud = get_pointcloud(d, m, pose)
    show_pointcloud(pointcloud)

    obj_frame = get_mjobj_frame(model=m, data=d, obj_name=pose) * sm.SE3.Rx(-np.pi)  # Get body frame
    obj_drop_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_box")  * sm.SE3.Rx(np.pi)# Get body frame
    pick_zone_frame = get_mjobj_frame(model=m, data=d, obj_name="pickup_point_cylinder") * sm.SE3.Rx(-np.pi) # Get body frame
    drop_zone_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_cylinder") *  sm.SE3.Rx(np.pi)  # Get body frame

    trajectory = via_points(robot, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame, steps=800)

    return trajectory