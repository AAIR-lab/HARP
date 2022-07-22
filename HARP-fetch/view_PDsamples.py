import numpy as np
from openravepy import *
import time
import sys
import os

sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
import fetch

'''
generate samples from the label distributions and plot them in openrave
'''

def sample_pd(pd):
    dof_sample = []
    for i in range(pd.shape[0]):
        print(pd[i,:])
        dof_bin = np.random.choice(range(number_of_bins), p=pd[i,:])
        print("sampled bin: ", dof_bin)
        dof_sample.append(dof_bin)

    print(dof_sample)
    return dof_sample

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
        dof_bin_range = dof_ranges[i]/number_of_bins
        s = llimits[i]
        for j in range(number_of_bins):
            dof_bins[i]['bin_start'].append(s)
            dof_bins[i]['bin_end'].append(s + dof_bin_range)
            s += dof_bin_range
    return dof_bins

def convert_sample(dof_sample):
    dof_values = []
    for i in range(len(dof_sample)):
        dof_value = (dof_bins[i]['bin_start'][dof_sample[i]] + dof_bins[i]['bin_end'][dof_sample[i]]) / 2.
        dof_values.append(dof_value)
    return dof_values



if __name__ == "__main__":
    envPath = 'envs/3d/7.1.dae'
    envnum = '7.1'
    runnum = 1

    number_of_samples = 1000
    number_of_bins = 10
    # pd_gt = np.load('data/env{envnum}/lbl/{envnum}.{runnum}.npy'.format(envnum=envnum, runnum=runnum))
    pd = np.load('results/env{envnum}.{runnum}/pd.npy'.format(envnum=envnum, runnum=runnum))
    pd = pd.reshape((8,10))
    inp = np.load('data/env{envnum}/inp/{envnum}.{runnum}.npy'.format(envnum=envnum, runnum=runnum))
    # inp = np.load('datatest/{envnum}.{runnum}.npy'.format(envnum=envnum, runnum=runnum))

    env = Environment()
    env.Load(envPath)
    env.SetViewer('qtcoin')
    initial_transform = pickle.load(open(os.path.join('envs','3d','fetch_transforms', envnum + '.pkl'), 'rb'))
    fetch_robot = fetch.FetchRobot(env, initial_transform)
    robot = env.GetRobots()[0]
    n_bin = 10
    bounds_dict = {
        '7.0': [[-2.5, -2.5, -0.001], [2.5, 2.5, 2.5]],
        '7.1': [[0, 0, -.1], [3., 4., 2.5]]
    }
    bounds = bounds_dict[envnum]
    trace = []

    llimits = robot.GetDOFLimits()[0]
    ulimits = robot.GetDOFLimits()[1]
    llimits[2] = llimits[12] = llimits[14] = -np.pi
    ulimits[2] = ulimits[12] = ulimits[14] = np.pi
    robot.SetDOFLimits(llimits, ulimits)

    dof_bins = get_dof_bins()


    # Denormalize dof values and visualize goal as green pixel
    goal = inp[0,0,0,3:11]
    llimits = robot.GetActiveDOFLimits()[0]
    ulimits = robot.GetActiveDOFLimits()[1]
    dof_ranges = ulimits - llimits
    denorm_goal = []
    for i, dof_value in enumerate(goal):
        denorm_dof = (dof_value * dof_ranges[i]) + llimits[i]
        denorm_goal.append(denorm_dof)
    robot.SetActiveDOFValues(denorm_goal)
    t = robot.GetManipulators()[0].GetTransform()
    trace.append(env.plot3(points=[t[0,3], t[1,3], t[2,3]], pointsize=0.03, colors=np.array((0,1,0)), drawstyle=1))
        
    collision_counts = 0
    for i in range(number_of_samples):
        dof_sample = sample_pd(pd)
        dof_values = convert_sample(dof_sample)
        robot.SetActiveDOFValues(dof_values)
        t = robot.GetManipulators()[0].GetTransform()

        if not env.CheckCollision(robot) and not robot.CheckSelfCollision():
            # visualize non colliding samples as red pixels
            trace.append(env.plot3(points=[t[0,3], t[1,3], t[2,3]], pointsize=0.03, colors=np.array((1,0,0)), drawstyle=1))
        else:
            collision_counts += 1
            # visualize collisions as blue pixels
            trace.append(env.plot3(points=[t[0,3], t[1,3], t[2,3]], pointsize=0.03, colors=np.array((0,0,1)), drawstyle=1))
        # time.sleep(2)
    print("Colliding samples: ", collision_counts)
    raw_input("Hit enter to exit")