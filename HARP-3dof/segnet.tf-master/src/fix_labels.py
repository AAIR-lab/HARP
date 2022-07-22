import os
import numpy as np
import matplotlib.pyplot as plt
import time

'''
Convert labels normalized to [-1,1] and created for regression to classification labels
inputs normalized to [0,1]
output pixels belong to class labels 
channel1(cr): [0-1]
channel2(dof1): [0-7]
channel3(dof2): [0-8] 
'''

def get_bin(value, dof_num):
    if dof_num == 1:
        d1_bin_start = [    -np.pi, -7*np.pi/8, -5*np.pi/8, -3*np.pi/8, -np.pi/8,   np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8]
        d1_bin_end   = [-7*np.pi/8, -5*np.pi/8, -3*np.pi/8,   -np.pi/8,  np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8,     np.pi]
        # d1_bin_value = [     np.pi, -3*np.pi/4,   -np.pi/2,   -np.pi/4,        0,   np.pi/4,   np.pi/2, 3*np.pi/4,     np.pi]
        d1_bin_value = [0, 1, 2, 3, 4, 5, 6, 7, 0]
        for i in range(len(d1_bin_start)):
            if value >= d1_bin_start[i] and value <= d1_bin_end[i]:
                return d1_bin_value[i]
    elif dof_num == 2:
        d2_bin_start = [-9*np.pi/16, -7*np.pi/16, -5*np.pi/16, -3*np.pi/16, -np.pi/16,   np.pi/16, 3*np.pi/16, 5*np.pi/16, 7*np.pi/16]
        d2_bin_end   = [-7*np.pi/16, -5*np.pi/16, -3*np.pi/16,   -np.pi/16,  np.pi/16, 3*np.pi/16, 5*np.pi/16, 7*np.pi/16, 9*np.pi/16]
        # d2_bin_value = [   -np.pi/2,  -3*np.pi/8,    -np.pi/4,    -np.pi/8,         0,    np.pi/8,    np.pi/4,  3*np.pi/8,    np.pi/2]
        d2_bin_value = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(len(d2_bin_start)):
            if value >= d2_bin_start[i] and value <= d2_bin_end[i]:
                return d2_bin_value[i]

if __name__ == "__main__":
    starttime = time.time()
    dataPath = 'input/raw'
    envlist = sorted(os.listdir(dataPath))
    for category in ['train', 'train-labels']:
        path = os.path.join(dataPath, category)
        samplefiles = sorted(os.listdir(path))
        for sample in samplefiles: 
            print("processing sample: {category}/{sample}".format(category=category, sample=sample))

            if sample.endswith('.npy') and category == 'train': # augment input
                inpfilePath = os.path.join(path, sample)
                inp = np.load(inpfilePath)

                env_tensor = inp[:,:,:3]
                goal_tensor = inp[:,:,3:]

                goal = goal_tensor[0,0]
                # goalx = (goal[0] * (5)) + (-2.5) # de normalize
                # goalx = 2*( (goalx - (-2.5)) / 5) - 1
                # de-norm
                goalx = ((goal[0]+1) * (2.5)) + (-2.5)
                goaly = ((goal[1]+1) * (2.5)) + (-2.5)
                # norm
                goalx = (goalx - (-2.5)) / (5.)
                goaly = (goaly - (-2.5)) / (5.)
                # goaly = (goal[1] * (5)) + (-2.5) # de normalize
                # goaly = 2*( (goaly - (-2.5)) / 5) - 1

                # denorm_d1 = (goal[2] * (2 * np.pi)) + (-np.pi)
                # goal_dof1 = 2*( (denorm_d1 - (-np.pi)) / (2*np.pi)) - 1
                denorm_d1 = ((goal[2]+1) * (np.pi)) + (-np.pi) # [-1,1]
                goal_dof1 = (denorm_d1 - (-np.pi)) / (2*np.pi)

                # denorm_d2 = (goal[3] * np.pi) + (-np.pi/2)
                # goal_dof2 = 2*( (denorm_d2 - (-np.pi/2)) / np.pi) - 1
                denorm_d2 = ((goal[3]+1) * (np.pi/2)) + (-np.pi/2) # [-1, 1]
                goal_dof2 = (denorm_d2 - (-np.pi/2)) / (np.pi) # [0,1]

                goal_tensor[:,:,0] = goalx
                goal_tensor[:,:,1] = goaly
                goal_tensor[:,:,2] = goal_dof1
                goal_tensor[:,:,3] = goal_dof2

                # mask = (env_tensor[:,:,0] == 0)
                # env_tensor[mask] = -1
                # env_tensor[~mask] = 1
                mask = (env_tensor[:,:,0] == -1)
                env_tensor[mask] = 0
                env_tensor[~mask] = 1            

                inp_tensor = np.concatenate([env_tensor, goal_tensor], axis=2)

                assert np.sum(np.isnan(inp_tensor)) == 0
                assert len(inp_tensor.shape) == 3
                assert inp_tensor.shape[2] == 7 # should have 7 channels (0-2: env image, 3-7: goal dof values)
                for x in range(inp_tensor.shape[2]): # all channels should be normalized
                    assert inp_tensor[:,:,x].min() >= 0.0 and inp_tensor[:,:,x].max() <= 1.0

                newpath = os.path.join(dataPath, category+'_cls', sample)
                np.save(newpath, inp_tensor)

            elif sample.endswith('.npy') and category == 'train-labels': # augment labels
                lblfilePath = os.path.join(path, sample)
                lbl = np.load(lblfilePath)

                cr_channel = lbl[:,:,0]
                # mask = (cr_channel[:,:] == 0)
                # cr_channel[mask] = -1
                # cr_channel[~mask] = 1
                mask = (cr_channel[:,:] == -1)
                cr_channel[mask] = 0
                cr_channel[~mask] = 1
                cr_channel = cr_channel.reshape(224,224,1)

                dof1_channel = lbl[:,:,1]
                for x in range(dof1_channel.shape[0]):
                    for y in range(dof1_channel.shape[1]):
                        # denorm_d1 = (dof1_channel[x,y] * (2 * np.pi)) + (-np.pi)
                        # goal_dof1 = 2*( (denorm_d1 - (-np.pi)) / (2*np.pi)) - 1
                        denorm_d1 = ((dof1_channel[x,y]+1) * (np.pi)) + (-np.pi) # [-1,1]
                        bin_ = get_bin(denorm_d1, 1)
                        dof1_channel[x,y] = bin_
                dof1_channel[mask] = 8 # UNK region label
                dof1_channel = dof1_channel.reshape(224,224,1)

                dof2_channel = lbl[:,:,2]
                for x in range(dof2_channel.shape[0]):
                    for y in range(dof2_channel.shape[1]):
                        # denorm_d2 = (dof2_channel[x,y] * (2 * np.pi)) + (-np.pi)
                        # goal_dof2 = 2*( (denorm_d2 - (-np.pi)) / (2*np.pi)) - 1
                        denorm_d2 = ((dof2_channel[x,y]+1) * (np.pi/2)) + (-np.pi/2) # [-1, 1]
                        bin_ = get_bin(denorm_d2, 2)
                        dof2_channel[x,y] = bin_
                dof2_channel[mask] = 9 # UNK region label
                dof2_channel = dof2_channel.reshape(224,224,1)

                lbl_tensor = np.concatenate([cr_channel, dof1_channel, dof2_channel], axis=2)

                assert np.sum(np.isnan(lbl_tensor)) == 0
                assert len(lbl_tensor.shape) == 3
                assert lbl_tensor.shape[2] == 3 # should have 3 channels (0: critical region gaze, 1-2: dof values)
                # for x in range(lbl_tensor.shape[2]): # all channels should be normalized
                assert lbl_tensor[:,:,0].min() >= 0 and lbl_tensor[:,:,0].max() <= 1.0
                assert lbl_tensor[:,:,1].min() >= 0 and lbl_tensor[:,:,1].max() <= 8
                assert lbl_tensor[:,:,2].min() >= 0 and lbl_tensor[:,:,2].max() <= 9

                newpath = os.path.join(dataPath, category+'_cls', sample)
                np.save(newpath, lbl_tensor)
    print("Completed in {runtime}mins".format(runtime=(time.time()-starttime)/60.0))