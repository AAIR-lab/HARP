import pdb
import time
import numpy as np
import os
from sys import *
from numpy import *
from openravepy import *
from trainingdataLbot import mNavigationPlanningLabels
import pickle

numbodies = 0 #int(argv[1])
envnum = 5.1 #float(argv[2])
runnum = 0 #int(argv[3])

np.random.seed(None)
st0 = np.random.get_state()
#pdb.set_trace()


env = Environment()
collisionChecker = RaveCreateCollisionChecker(env, 'fcl_')
collisionChecker.SetCollisionOptions(CollisionOptions.ActiveDOFs)
env.SetCollisionChecker(collisionChecker)
env.SetViewer('qtcoin')

# load and set environment
env.Load('envs3d/' + str(envnum) + '.xml')
# load robots
robot = env.GetRobots()[0]

trajectory_info = pickle.load(open('data/env5.1/data/5.1.0_traj.pkl', 'rb'))
for trajectory in trajectory_info:
    print(trajectory['start'])
    print(trajectory['goal'])
    print(len(trajectory['path']))
    traj = RaveCreateTrajectory(env, '')
    Trajectory.deserialize(traj, trajectory['traj'])
    with robot:
        robot.GetController().SetPath(traj)
    robot.WaitForController(0)