import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from openravepy import *
import time
import multiprocessing as mp
import sys
import os
import random

sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
import fetch

'''
visualize channels and input in openrave
'''


def getpixelcoords(i, j, k):
    pixminx = bounds[0][0] + (i * pdims[0])
    pixmaxx = bounds[0][0] + ((i + 1) * pdims[0])

    pixminy = bounds[0][1] + (j * pdims[1])
    pixmaxy = bounds[0][1] + ((j + 1) * pdims[1])
    
    pixminz = bounds[0][2] + ((k + 1) * pdims[2])
    pixmaxz = bounds[0][2] + (k * pdims[2])
    return [(pixminx + pixmaxx)/2, (pixminy + pixmaxy)/2, (pixminz + pixmaxz)/2]

def get_dof_bins():
    #create dof bins
    llimits = robot.GetActiveDOFLimits()[0]
    ulimits = robot.GetActiveDOFLimits()[1]
    dof_ranges = ulimits - llimits
    dof_bins = {}
    for i in range(len(dof_ranges)):
        dof_bins[i] = {}
        dof_bins[i]['bin_start'] = []
        dof_bins[i]['bin_end'] = []
        dof_bin_range = dof_ranges[i]/n_bin
        s = llimits[i]
        for j in range(n_bin):
            dof_bins[i]['bin_start'].append(s)
            dof_bins[i]['bin_end'].append(s + dof_bin_range)
            s += dof_bin_range
    return dof_bins

def view_inp(inp):
    assert inp.shape[-1] == 17
    for i in xrange(labelsize):
        for j in xrange(labelsize):
            for k in xrange(labelsize):
                if inp[i,j,k,0] == 0:
                    q = getpixelcoords(i, j, k)
                    trace.append(env.plot3(points=[q[0], q[1], q[2]], pointsize=0.03, colors=np.array((1,0,0)), drawstyle=1))
    return

def view_channels(inp, threshold):
    assert inp.shape[-1] == 9
    u_counts = 0
    for i in xrange(labelsize):
        for j in xrange(labelsize):
            for k in xrange(labelsize):
                if inp[i,j,k,0] >= threshold:
                    q = getpixelcoords(i, j, k)
                    pl = trace.append(env.plot3(points=[q[0], q[1], q[2]], pointsize=0.03, colors=np.array((1,0,0)), drawstyle=1))

                    dof_vals = inp[i,j,k,1:]
                    dof_converted = []
                    unknown = False
                    for dof_index in dof_bins.keys():
                        bin_index = dof_vals[dof_index]
                        if bin_index == 11:
                            unknown = True
                            u_counts += 1
                            break
                        # sample range of bin uniformly
                        # dof_value = random.uniform(dof_bins[int(dof_index)]['bin_start'][int(bin_index)], dof_bins[int(dof_index)]['bin_end'][int(bin_index)])
                        
                        # use average of bin range
                        dof_value = (dof_bins[int(dof_index)]['bin_start'][int(bin_index)] + \
                                    dof_bins[int(dof_index)]['bin_end'][int(bin_index)]) / 2.
                        dof_converted.append(dof_value)
                    if not unknown:
                        robot.SetActiveDOFValues(dof_converted)

                        # body = RaveCreateKinBody(env, '')
                        # body.SetName('tracer')
                        # body.InitFromBoxes(np.array([[q[0], q[1], q[2], pdims[0], pdims[1], pdims[2]]]), True)
                        # env.AddKinBody(body)
                        # env.UpdatePublishedBodies()
                        # env.Remove(body)
    print(u_counts)
    return  

if __name__ == "__main__":
    envPath = 'envs/3d/7.0.dae'
    envnum = '7.0'
    runnum = 1

    env = Environment()
    env.Load(envPath)
    env.SetViewer('qtcoin')
    initial_transform = pickle.load(open(os.path.join('envs','3d','fetch_transforms', str(envnum) + '.pkl'), 'rb'))
    fetch_robot = fetch.FetchRobot(env, initial_transform)
    # robot = env.GetRobots()[0]
    n_bin = 10
    trace = []
    labelsize = 64
    # bounds_dict = {
    #     '7.0': [[-2.5, -2.5, -0.001], [2.5, 2.5, 2.5]],
    #     '7.1': [[0, 0, -.1], [3., 4., 2.5]]
    # }
    robot = env.GetRobots()[0]
    robot_traform = robot.GetTransform()
    # bounds = [[-1, -1, 0], [1, 1, 2]]
    bounds = [[robot_traform[0,3]-1,robot_traform[1,3]-1,0],[robot_traform[0,3]+1,robot_traform[1,3]+1,2.0]]
    pdims = [0,0,0]
    pdims[0] = (bounds[1][0] - bounds[0][0]) / labelsize # x
    pdims[1] = (bounds[1][1] - bounds[0][1]) / labelsize # y
    pdims[2] = (bounds[1][2] - bounds[0][2]) / labelsize # z


    llimits = robot.GetDOFLimits()[0]
    ulimits = robot.GetDOFLimits()[1]
    llimits[2] = llimits[12] = llimits[14] = -np.pi
    ulimits[2] = ulimits[12] = ulimits[14] = np.pi
    robot.SetDOFLimits(llimits, ulimits)

    dof_bins = get_dof_bins()

    inp = np.load("./network/results/{}..npy")
    # inp = np.load('data/env{envnum}/inp/{envnum}.{runnum}.npy'.format(envnum=envnum, runnum=runnum))
    # inp = np.load('datatest/{envnum}.{runnum}.npy'.format(envnum=envnum, runnum=runnum))

    # Plot pixels in OR
    view_inp(inp)
    # view_channels(inp, 1) # for channel based classification tasks

    # Denormalize dof values
    goal = inp[0,0,0,3:11]
    llimits = robot.GetActiveDOFLimits()[0]
    ulimits = robot.GetActiveDOFLimits()[1]
    dof_ranges = ulimits - llimits
    denorm_goal = []
    for i, dof_value in enumerate(goal):
        denorm_dof = (dof_value * dof_ranges[i]) + llimits[i]
        denorm_goal.append(denorm_dof)
    print("Goal Config: {goal}".format(goal=denorm_goal))

    # Denormalize fetch location
    fetch_loc = inp[0,0,0,11:]
    fetch_xyz = fetch_loc[0:3]
    fetch_axisAngles = fetch_loc[3:]
    denorm_xyz = []
    for i in range(3):
        denorm_xyz.append((fetch_xyz[i] * (bounds[1][i] - bounds[0][i])) + (bounds[0][i]))
    denorm_axisAngles = []
    for i in range(3):
        denorm_axisAngles.append((fetch_axisAngles[i] * (2*np.pi)) + (-np.pi))

    # Create updated fetch location's transform matrix
    t = matrixFromAxisAngle(denorm_axisAngles)
    t[0,3] = denorm_xyz[0]
    t[1,3] = denorm_xyz[1]
    t[2,3] = denorm_xyz[2]

    # Check goal in current view
    current_dof_vals = robot.GetActiveDOFValues()
    robot.SetActiveDOFValues(denorm_goal)
    time.sleep(1)
    robot.SetActiveDOFValues(current_dof_vals)
    time.sleep(1)

    # Move to augmented position
    robot.SetTransform(t)

    # Check goal in augmented location
    robot.SetActiveDOFValues(denorm_goal)

    raw_input("Press Enter to exit")