import pdb, time
import sys
import ompl.app as app
from ompl import base as ob
from ompl import geometric as og
from math import pi
from openravepy import *
from numpy import *
from functools import partial
import numpy as np
import os
import pickle

sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
import fetch

test_sample = sys.argv[1]
numruns = int(sys.argv[2])
MAXTIME = int(sys.argv[3])
planner = sys.argv[4]

# test_sample = '7.2.1'
# numruns = 10
# MAXTIME = 20
# planner = 'birrt'

envnum = test_sample[:-2]
visualize = False

env = None
robot = None
jointLimits = []
trace = []

def isStateValid(si, state):
    global env
    global robot
    valid = True
    if si.satisfiesBounds(state):
        values = []
        for i in xrange(8):
            values.append(state[i][0])

        # with env:
        robot.SetActiveDOFValues(values)
        report = CollisionReport()
        if env.CheckCollision(robot, report) or robot.CheckSelfCollision(report):
            valid = False
        else:
            if visualize:
                t = robot.GetManipulators()[0].GetTransform()
                trace.append(env.plot3(points=[t[0,3], t[1,3], t[2,3]], pointsize=0.03, colors=np.array((1,0,0)), drawstyle=1))
    else:
        valid = False

    return valid

def plan(s, g, plannertype, jointlimits):
    # construct compound state space
    statespace = ob.CompoundStateSpace()

    for i in range(len(jointlimits[0])):
        joint_vec_space = ob.RealVectorStateSpace(1)
        joint_vec_bounds = ob.RealVectorBounds(1)
        joint_vec_bounds.setLow(jointlimits[0][i])
        joint_vec_bounds.setHigh(jointlimits[1][i])
        joint_vec_space.setBounds(joint_vec_bounds)
        statespace.addSubspace(joint_vec_space, 1.0)

    # set up motion plan
    ss = og.SimpleSetup(statespace)
    si = ss.getSpaceInformation()
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(isStateValid, si)))
    
    # load start and goal
    start = ob.State(statespace)
    for i, val in enumerate(s):
        start[i] = val
    goal = ob.State(statespace)
    for i, val in enumerate(g):
        goal[i] = val
    ss.setStartAndGoalStates(start, goal)

    # set up planner
    if plannertype == 'birrt':
        planner = og.RRTConnect(si)
        planner.setRange(0.2)
    elif plannertype == 'rrt':
        planner = og.RRT(si)
        planner.setRange(0.2)
    elif plannertype == 'prm':
        planner = og.PRM(si)
    
    ss.setPlanner(planner)
    ss.setup()

    if plannertype == 'prm':
        planner.growRoadmap(1)

    # execute planner
    solved = ss.solve(MAXTIME)
    if solved and ss.haveExactSolutionPath():
        print 'Length: ' + str(ss.getSolutionPath().length())
        #print("Found solution:\n%s" % ss.getSolutionPath())
        return True 
    else:
        return False

def setupRobot(robot):
    if robot.GetName() == 'robot1':
        robot.SetDOFVelocities([0.5,0.5,0.5])
        robot.SetActiveDOFs([0,1,2])
        robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
    elif robot.GetName() == 'lshaped':
        robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
        robot.SetActiveDOFs([0,1,2,4])
        robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])
    elif robot.GetName() == 'fetch':
        llimits = robot.GetDOFLimits()[0]
        ulimits = robot.GetDOFLimits()[1]
        # DOFs 2, 12, 14 are circular and need their limits to be set to [-3.14, 3.14] as stated by planner
        llimits[2] = llimits[12] = llimits[14] = -np.pi
        ulimits[2] = ulimits[12] = ulimits[14] = np.pi
        robot.SetDOFLimits(llimits, ulimits)

def main():
    global env
    global robot
    global jointLimits

    if not os.path.isdir(os.path.join('results', 'env' + test_sample)):
        os.makedirs(os.path.join('results', 'env' + test_sample))
    results_file = open(os.path.join('results', 'env' + test_sample, 'plan_' + planner.lower() + '.csv'), 'a')

    env = Environment()
    if visualize: 
        env.SetViewer('qtcoin')
    env.Load('envs/3d/'+envnum+'.dae')

    initial_transform = pickle.load(open(os.path.join('envs','3d','fetch_transforms', str(envnum) + '.pkl'), 'rb'))
    fetch_robot = fetch.FetchRobot(env, initial_transform)

    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    env.SetCollisionChecker(collisionChecker)
    robot = env.GetRobots()[0]

    setupRobot(robot)

    jointlimits = list(robot.GetActiveDOFLimits())
    env.UpdatePublishedBodies()

    # tuck pose
    start = [ 0.00000000e+00,  1.32000492e+00,  1.39998343e+00, -1.99845334e-01,
    1.71996798e+00,  2.70622649e-06,  1.66000647e+00, -1.34775426e-06]

    goals = { 
        '7.0.1':  [ 0.15317293, -0.02241246,  0.3       ,  2.8       , -0.17657254, 1.89736047, -1.64968682,  1.10353645],
         '8.0.1':  [ 0.2194822 ,  0.68207005,  0.4       , -0.8       ,  1.86541826, 0.71582408, -1.54721034,  1.04590355]
    }
    goal = goals[test_sample]
    
    success_count = 0
    for i in xrange(numruns):
        starttime = time.time()
        success = plan(start, goal, planner, jointlimits)
        print("Completed in {total_time}".format(total_time=(time.time() - starttime)))
        if success:
            success_count += 1
        print(success, success_count, i+1, planner, MAXTIME)
    print(success_count, planner, MAXTIME)
    results_file.write(str(MAXTIME) + ',' + str(success_count) + '\n')
    results_file.close()
    
    RaveDestroy()

if __name__ == "__main__":
    main()