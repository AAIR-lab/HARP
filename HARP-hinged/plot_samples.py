from __future__ import division
from openravepy import *
from functools import partial
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from sys import *
from numpy import * 
import numpy
import numpy as np
import time
import os
from heapq import *
import pdb 
import pickle

RaveSetDebugLevel(0)

# envnum = argv[1]
# NUMRUNS = int(argv[2])
# MAXTIME = float(argv[3])
# mode = argv[4].lower()

test_sample = '10.0.2'
envnum = test_sample[:-2]

def load_samples():
    with open(os.path.join('results', 'env' + test_sample, 'samples.pkl'), 'rb') as samplefile:
        samples = pickle.load(samplefile)
    print("Total samples: ", len(samples))
    return samples

def collides(q):
    collision = False
    robot = env.GetRobots()[0]	

    with env:
        prev = robot.GetActiveDOFValues()
        robot.SetActiveDOFValues(q)
        if env.CheckCollision(robot):
            collision = True
        robot.SetActiveDOFValues(prev)

    return collision

env = Environment()
env.Load('envs3d/'+envnum+'.xml')

env.SetViewer('qtcoin')
collisionChecker = RaveCreateCollisionChecker(env, 'fcl_')
env.SetCollisionChecker(collisionChecker)

# load robots
robot = env.GetRobots()[0]
robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
robot.SetActiveDOFs([0,1,2,4])
robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])

samples = load_samples()
trace = []
for i in range(len(samples)):
    randsample = np.random.choice(len(samples), size=1, replace=False)
    # minx, miny, maxx, maxy, [dof1, dof2]
    pointx = (samples[i][0]+samples[i][2])/2.0
    pointy = (samples[i][1]+samples[i][3])/2.0
    dof1 = samples[i][4][0]
    dof2 = samples[i][4][1]
    q = [pointx, pointy, dof1, dof2]
    robot.SetActiveDOFValues(q)
    if not collides(q):
        trace.append(env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((1,0,0)), drawstyle=1))

x = raw_input("Press enter to exit")