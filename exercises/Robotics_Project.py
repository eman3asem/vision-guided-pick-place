import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import random
import mujoco

from ompl import base as ob
from ompl import geometric as og

from robot import *

Close_gripper=255
Open_gripper=0

def points(robot, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame):

    via_points_list = []
    q0 = robot.get_current_q()
    via_points_list.append((q0,Open_gripper))

    for i in range(len(obj_frame)):
        print(i)
        if i == 2: # cylinder, remember to change this if you change the order of the objects
            z_offset = sm.SE3.Tz(-0.09)
        else:
            z_offset = sm.SE3.Tz(-0.02)
        #1
        q_start_zone=robot.robot_ur5.ik_LM(Tep=pick_zone_frame *sm.SE3.Tz(-0.15) , q0=q0)[0]
        via_points_list.append((q_start_zone,Open_gripper))

        #2
        q_pre_pick = robot.robot_ur5.ik_LM(Tep=obj_frame[i]* sm.SE3.Tz(-0.15), q0=q_start_zone)[0]
        via_points_list.append((q_pre_pick, Open_gripper))

        #3
        q_pick = robot.robot_ur5.ik_LM(Tep=obj_frame[i]* z_offset, q0=q_pre_pick)[0]
        via_points_list.append((q_pick,Close_gripper))

        #4
        q_after_pick = robot.robot_ur5.ik_LM(Tep=obj_frame[i] * sm.SE3.Tz(-0.15), q0=q_pick)[0]
        via_points_list.append((q_after_pick, Close_gripper))

        #5
        q_start_zone=robot.robot_ur5.ik_LM(Tep=pick_zone_frame, q0=q_after_pick)[0]
        via_points_list.append((q_start_zone, Close_gripper))

        #6
        q_drop_zone = robot.robot_ur5.ik_LM(Tep=drop_zone_frame, q0=q_start_zone)[0]
        via_points_list.append((q_drop_zone, Close_gripper))

        #7
        q_pre_drop = robot.robot_ur5.ik_LM(Tep=obj_drop_frame[i] * sm.SE3.Tz(0.15), q0=q_drop_zone)[0]
        via_points_list.append((q_pre_drop, Open_gripper))

        #8
        q_drop_zone = robot.robot_ur5.ik_LM(Tep=drop_zone_frame, q0=q_pre_drop)[0]
        via_points_list.append((q_drop_zone, Open_gripper))
        q0 = q_drop_zone

    return via_points_list


## ======= excercise 4 code below ======= ##
def parabolic_q_interpolation(start_q, end_q, steps):
    seg = []
    q0 = start_q[0]
    qf = end_q[0]
    t0 = 0
    tf = 3
    td = tf-t0 # duration of travel
    tb = tf * 0.3 # blend time
    steps = steps

    # Calculate required acceleration for the blend
    ddqb = (qf - q0) / (tb * (td - tb))  # constant acceleration during blend
    # print(ddqb)

    for t in np.linspace(t0, tf, steps):
        if t0 <= t and t < t0 + tb:
            q_t = q0 + 0.5 * ddqb*(t-t0)**2
        elif t0 + tb <= t and t < tf - tb:
            q_t = q0 + 0.5*ddqb*tb**2 + ddqb*tb*(t - t0 - tb)
        elif  tf-tb <= t and t < tf:
            q_t = qf - 0.5*ddqb*(tf-t)**2
        seg.append((q_t, start_q[1]))
    return seg
## ======= excercise 4 code end ======= ##

## ======= excercise 5 code below ======= ##
def via_points(robot, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame, steps):
    print("Generating P2P trajectory")
    via_q= points(robot, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame)
    P2P_trajectory = []
    for i in range(1, len(via_q)):
        P2P_trajectory.extend(parabolic_q_interpolation(start_q=via_q[i-1], end_q=via_q[i], steps=steps))
    return P2P_trajectory
## ======= excercise 5 code end ======= ##

## ======= excercise 6 code below ======= ##
class StateValidator:
    # This is mujoco specific, so I have implemented this for you
    def __init__(self, d, m, num_joint):
        self.d = d
        self.m = m
        self.num_joint = num_joint
    
    def __call__(self, state):
        # print("isStateValid - state: ", state)
        q_pose = [state[i] for i in range(self.num_joint)]
        return is_q_valid(d=self.d, m=self.m, q=q_pose) 

# RRT planner function
def RRT_planner(d, m, start_q, goal_q):
    num_joint = 6
    space = ob.RealVectorStateSpace(num_joint) # Create a joint-space vector instead of a 2D space as in the example
    # Create joint bounds
    bounds = ob.RealVectorBounds(num_joint)
    bounds.setLow(-3.2)
    bounds.setHigh(3.2)
    space.setBounds(bounds)
    
    # Create SimpleSetup
    ss = og.SimpleSetup(space)
    validator = StateValidator(d, m, num_joint)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
    
    # Set start and goal states
    start = ob.State(space)
    goal = ob.State(space)
    
    # Set specific joint values for start and goal
    for i in range(num_joint):
        start[i] = start_q[i]  # initial joint angles
        goal[i] = goal_q[i]    # goal joint angles
    
    ss.setStartAndGoalStates(start, goal)

    planner = og.RRT(ss.getSpaceInformation())

    ss.setPlanner(planner)
    
    # Solve the problem
    solved = ss.solve(5.0)
    
    if solved:
        # Get the solution path
        path = ss.getSolutionPath()
        print("Found a Solution!")
        # Print basic information about the path
        print(f"Path length: {path.length()}")
        print(f"Number of states: {path.getStateCount()}")

        RRT_trajectory = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q_pose = [state[i] for i in range(space.getDimension())]
            # print(f"State {i}: {q_pose}")
            RRT_trajectory.append(np.array(q_pose))

        return RRT_trajectory
    else: 
        print("No Solution Found!")
        return None
## ======= excercise 6 code end ======= ##

def RRT(robot, d, m, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame):
    EXE_TIME= 1000 #ms
    q0 = robot.get_current_q()
    # Move to pick zone 
    goal_q_start = robot.robot_ur5.ik_LM(Tep=pick_zone_frame, q0=q0)[0]
    robot.move_j(start_q=q0, end_q=goal_q_start, t=EXE_TIME)

    for i in range(len(obj_frame)):
          
        if i == 2: # cylinder, remember to change this if you change the order of the objects
            z_offset = sm.SE3.Tz(-0.09)
        else:
            z_offset = sm.SE3.Tz(-0.02)
        # move to the object


        goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame[i]* z_offset, q0=robot.queue[-1][0])[0]
        sol_traj = RRT_planner(d=d, m=m, start_q=robot.queue[-1][0], goal_q=goal_q)
        robot.move_j_via(points=sol_traj, t=EXE_TIME)

        robot.set_gripper(255) # Close gripper

        goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame[i]* sm.SE3.Tz(-0.15), q0=robot.queue[-1][0])[0]
        sol_traj = RRT_planner(d=d, m=m, start_q=robot.queue[-1][0], goal_q=goal_q)
        robot.move_j_via(points=sol_traj, t=EXE_TIME)
    
        goal_q = robot.robot_ur5.ik_LM(Tep=obj_drop_frame[i]* sm.SE3.Tz(0.08), q0=robot.queue[-1][0])[0]
        sol_traj = RRT_planner(d=d, m=m, start_q=robot.queue[-1][0], goal_q=goal_q)
        robot.move_j_via(points=sol_traj, t=EXE_TIME)

        robot.set_gripper(0) # open gripper

    return robot.queue

# Hybrid P2P + RRT planner function for future work
def Hybrid_P2P_RRT(robot, d, m, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame):

    q0 = robot.get_current_q()
    # Move to pick zone 
    goal_q_start = robot.robot_ur5.ik_LM(Tep=pick_zone_frame, q0=q0)[0]
    robot.move_j(start_q=q0, end_q=goal_q_start, t=500)

    for i in range(len(obj_frame)):
        EXE_TIME = 500 #ms  
        # move to the object
        start_q = robot.get_current_q()
        goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame[i]* sm.SE3.Tz(-0.15), q0=start_q)[0]
        robot.move_j(start_q=start_q, end_q=goal_q, t=EXE_TIME)
        #back to pickup position
        goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame[i], q0=robot.queue[-1][0])[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=goal_q, t=EXE_TIME)
        
        robot.set_gripper(255) # Close gripper

        goal_q = robot.robot_ur5.ik_LM(Tep=obj_frame[i] * sm.SE3.Tz(-0.15), q0=robot.queue[-1][0])[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=goal_q, t=EXE_TIME)
        
        # Plan path to drop zone using RRT - need IK for goal
        goal_q_drop = robot.robot_ur5.ik_LM(Tep=drop_zone_frame, q0=robot.queue[-1][0])[0]
        sol_traj = RRT_planner(d=d, m=m, start_q=robot.queue[-1][0], goal_q=goal_q_drop)

        robot.move_j_via(points=sol_traj, t=1000)

        goal_q = robot.robot_ur5.ik_LM(Tep=obj_drop_frame[i] * sm.SE3.Tz(0.15), q0=robot.queue[-1][0])[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=goal_q, t=EXE_TIME)

        robot.set_gripper(0) # open gripper

        # Go up again - need IK
        goal_q_up = robot.robot_ur5.ik_LM(Tep=drop_zone_frame, q0=robot.queue[-1][0])[0]
        robot.move_j(start_q=robot.queue[-1][0], end_q=goal_q_up, t=EXE_TIME)

        # Plan path back to pick zone using RRT - need IK for goal
        goal_q_pick = robot.robot_ur5.ik_LM(Tep=pick_zone_frame, q0=robot.queue[-1][0])[0]
        sol_traj = RRT_planner(d=d, m=m, start_q=robot.queue[-1][0], goal_q=goal_q_pick)

        robot.move_j_via(points=sol_traj, t=EXE_TIME)
    return robot.queue
    


def program(d, m):
    # Define our robot object
    robot = UR5robot(data=d, model=m)

    box_frame = get_mjobj_frame(model=m, data=d, obj_name="box") * sm.SE3.Rx(-np.pi)  # Get body frame
    drop_point_box_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_cylinder") * sm.SE3.Rx(np.pi)  # Get body frame

    t_block_frame = get_mjobj_frame(model=m, data=d, obj_name="t_block") * sm.SE3.Rx(np.pi) *sm.SE3.Rz(np.pi/2) # Get body frame
    drop_point_t_block_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_tblock") * sm.SE3.Rx(-np.pi)*sm.SE3.Rz(np.pi/2) # Get body frame

    cylinder_frame = get_mjobj_frame(model=m, data=d, obj_name="cylinder") *sm.SE3.Rx(np.pi)*sm.SE3.Rz(np.pi/2) # Get body frame
    drop_point_cylinder_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_box") *sm.SE3.Rx(np.pi)*sm.SE3.Rz(np.pi/2) # Get body frame

    pick_zone_frame = get_mjobj_frame(model=m, data=d, obj_name="pickup_point_cylinder") * sm.SE3.Rx(-np.pi) # Get body frame
    drop_zone_frame = get_mjobj_frame(model=m, data=d, obj_name="drop_point_cylinder") *  sm.SE3.Rx(np.pi)  # Get body frame
    
    obj_frame = [t_block_frame, box_frame, cylinder_frame]
    obj_drop_frame = [drop_point_t_block_frame, drop_point_box_frame, drop_point_cylinder_frame]
    
    # ===== EXPLICITLY SET PLANNER from exercise 6 =====

    usr_input = input("Points/RRT: ")

    trajectory = []
    # P2P
    if usr_input.lower() == "points":
        print("Point to Point planner")
        trajectory = via_points(robot, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame, steps=800)
    # RRT
    elif usr_input.lower() == "rrt":
        print("RRT Planner")
        trajectory = RRT(robot, d, m, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame)
    #wrong input
    else:
        print("Wrong input, defaulting to RRT")
        trajectory = RRT(robot, d, m, obj_frame, obj_drop_frame, pick_zone_frame, drop_zone_frame)
        

    return trajectory