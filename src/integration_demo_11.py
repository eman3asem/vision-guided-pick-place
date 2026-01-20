import mujoco as mj
from spatialmath import SE3, SO3
from spatialmath.base import trnorm
from scipy.spatial.transform import Rotation
import math
import random
import mujoco

from src.cam import *

from src.robot import *
from src.pose_estimation import do_pose_estimation

from src.cam import get_pointcloud, get_camera_pose_cv
from src.UR5_p2p_planner import via_points
from src.helpers import computeError


def r2q(rot):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]
    using scipy's Rotation class
    """
    r = Rotation.from_matrix(rot)
    return r.as_quat()  # Returns [x, y, z, w]

def program(d, m):
    # Computer vision
    # _width = 640*3,
    # _height = 480*3,
    # Initialize OpenGL context
    # mj.GLContext(max_width=640, max_height=480)
    # Create renderer

    camera_name = "cam1"

    # Random duck position
    rand_x = random.uniform(0.45,0.75)
    rand_y = random.uniform(-0.55,0.55)
    z = 0.025
    d.joint('duck').qpos [0:3]= [rand_x, rand_y, z+0.01]
    rand_rot = random.randint(0,359)
    rot = SO3.Eul(rand_rot, 90, 90,unit="deg").R
    d.joint('duck').qpos[3:] = r2q(rot)

    #The duck was falling, solution was to slow down the simulation a bit
    # 100 steps = 0.2 seconds of simulation time
    for _ in range(100):
        mujoco.mj_step(m, d)
    
    # get duck position after it has settled on the table
    duck_pos = d.body('duck').xpos
    duck_rot = d.body('duck').xmat.reshape(3, 3)
    duck_rot = trnorm(duck_rot)
    duck_se3 = SE3.Rt(duck_rot, duck_pos)

    # mujoco.mj_step(m, d)

    cam_se3_2 = get_camera_pose_cv(m, d, camera_name=camera_name)

    gt = cam_se3_2.inv() * duck_se3

    id = 0

    with open(f"gt_{id:04}.txt", 'w') as f:
        for i in range(4):
            for j in range(4):
                f.write(f"{gt.A[i,j]} ")
            f.write("\n")
    
    renderer = mj.Renderer(m, height=480, width=640)
    get_pointcloud(m, d, renderer, f"point_cloud_{id:04}.pcd", camera_name=camera_name)
    # show_pointcloud(f"point_cloud_{id:04}.pcd")


    #################### my code starts here #######################
    robot = UR5robot(data=d, model=m)
    # Pose estimate the duck
    scene_pointcloud = o3d.io.read_point_cloud(f"point_cloud_{id:04}.pcd")
    #==============================================#
    #from trail_run.py file in Vision project
    # Load duck model as point cloud
    duck_mesh = o3d.io.read_triangle_mesh('./src/duck.stl') # remember to change the path to avoid errors
    duck_pointcloud = duck_mesh.sample_points_poisson_disk(10000)
    
    # Estimate duck pose in camera coordinates
    estimated_pose = do_pose_estimation(scene_pointcloud, duck_pointcloud)

    ground_truth = np.loadtxt(f"gt_{id:04}.txt")
    
    print("Ground truth")
    print(ground_truth)

    print("Error")
    print(computeError(ground_truth,estimated_pose))
    #==============================================#
    
    # Get camera pose
    cam_se3 = get_camera_pose_cv(m, d, camera_name="cam1")
    # Convert estimated pose to world coordinates
    duck_camera_pose = sm.SE3(estimated_pose, check=False)* sm.SE3.Rx(np.pi)  # Convert 4x4 numpy array to SE3
    # Transform to world frame
    duck_world_pose = cam_se3 * duck_camera_pose 
    # print(duck_world_pose) 
    
    
    # only choosing the traslation frame for grasping
    duck_position = duck_world_pose.t
    # adding a normal rotation to the grasp frame
    R_grasp = sm.SE3.Rx(-np.pi)

    #setting the frames for the Point-to-Point interpolator
    obj_frame = sm.SE3.Rt(R_grasp.R, duck_position) *sm.SE3.Rz(np.pi/2)
    obj_drop_frame = get_mjobj_frame(model=m, data=d, obj_name="zone_drop") *sm.SE3.Rx(-np.pi) * sm.SE3.Tz(-0.30)
    

    pick_zone_frame = obj_frame * sm.SE3.Tz(-0.15) # Pick zone
    drop_zone_frame = get_mjobj_frame(model=m, data=d, obj_name="zone_drop") *sm.SE3.Rx(-np.pi)*sm.SE3.Tz(-0.15) # Drop zone
    
    # getting the trajectory using Point-to-Point interpolator with trapezoidal velocity profile
    trajectory = via_points(robot, [obj_frame], [obj_drop_frame], pick_zone_frame, drop_zone_frame, steps=800)

    return trajectory