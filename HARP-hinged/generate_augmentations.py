import os
import numpy as np

'''
Processes generated data samples to create 4 new samples:
Rotate 90, 180, 270 and flip along horizontal axis
Output new labels appended with suffixes r1, r2, r3, f1
Calculates respective goal locations for new input samples
'''

def goal_rot(goal, rot_angle):
    # De-Normalize to [-2.5, 2.5]
    # [0,1]
    # goalx = (goal[0] * (5)) + (-2.5)
    # goaly = (goal[1] * (5)) + (-2.5)
    # [-1,1]
    goalx = ((goal[0]+1) * (2.5)) + (-2.5)
    goaly = ((goal[1]+1) * (2.5)) + (-2.5)

    rot_matrix = np.zeros((2,2))
    rot_matrix[0,0] = np.cos(rot_angle)
    rot_matrix[0,1] = -np.sin(rot_angle)
    rot_matrix[1,0] = np.sin(rot_angle)
    rot_matrix[1,1] = np.cos(rot_angle)
    loc_matrix = np.array([[goalx], [goaly]])
    updated_loc = np.matmul(rot_matrix, loc_matrix)

    new_goal = [0,0,0,0]
    # Normalize
    # [0,1]
    new_goal[0] = (updated_loc[0,0] - (-2.5)) / (5.)
    new_goal[1] = (updated_loc[1,0] - (-2.5)) / (5.)
    # [-1,1]
    # new_goal[0] = (2*(updated_loc[0,0] - (-2.5)) / (5.)) - 1
    # new_goal[1] = (2*(updated_loc[1,0] - (-2.5)) / (5.)) - 1

    # Denormalize to [-pi, pi]
    # denorm_d1 = (goal[2] * (2 * np.pi)) + (-np.pi) # [0,1]
    denorm_d1 = ((goal[2]+1) * (np.pi)) + (-np.pi) # [-1,1]
    new_goal[2] = denorm_d1 + rot_angle
    if new_goal[2] > np.pi:
        new_goal[2] = (new_goal[2] - np.pi) + (-np.pi)
    # normalize 
    # [0, 1]
    new_goal[2] = (new_goal[2] - (-np.pi)) / (2*np.pi)
    # [-1,1]
    # new_goal[2] = (2*(new_goal[2] - (-np.pi)) / (2*np.pi)) - 1
    new_goal[3] = goal[3]


    return new_goal

def goal_udflip(goal):
    new_goal = [0,0,0,0]
    # X,Y
    # De-Normalize
    # [0,1]
    # goalx = (goal[0] * (5)) + (-2.5) 
    # goaly = (goal[1] * (5)) + (-2.5)
    # [-1,1]
    goalx = ((goal[0]+1) * (2.5)) + (-2.5)
    goaly = ((goal[1]+1) * (2.5)) + (-2.5)

    # x is unchanged, y is flipped
    goalx = goalx
    goaly = -goaly

    # Normalize
    # [0,1]
    # new_goal[0] = (goalx - (-2.5)) / (5.)
    # new_goal[1] = (goaly - (-2.5)) / (5.)
    # [-1,1]
    new_goal[0] = (2*(goalx - (-2.5)) / (5.)) - 1
    new_goal[1] = (2*(goaly - (-2.5)) / (5.)) - 1

    # DOF 1
    # De-Normalize
    # denorm_d1 = (goal[2] * (2 * np.pi)) + (-np.pi) # [0,1]
    denorm_d1 = ((goal[2]+1) * (np.pi)) + (-np.pi) # [-1,1]
    
    # angle is flipped
    goal_d1 = -denorm_d1
    
    # Normalize
    d1_val_norm = (goal_d1 - (-np.pi)) / (2*np.pi) # [0,1]
    # d1_val_norm = (2*(goal_d1 - (-np.pi)) / (2*np.pi)) - 1 # [-1,1]
    new_goal[2] = d1_val_norm
    
    # DOF 2
    # De-Normalize
    denorm_d2 = (goal[3] * np.pi) + (-np.pi/2) # [0,1]
    # denorm_d2 = ((goal[3]+1) * (np.pi/2)) + (-np.pi/2) # [-1, 1]
    denorm_d2 = -denorm_d2
    denorm_d2 = (denorm_d2 - (-np.pi) / (2 * np.pi))

    new_goal[3] = denorm_d2
    
    return new_goal

def get_dof_bins():
    #create dof bins
    uulimit = np.pi 
    llimit = -np.pi
    number_of_bins = 10
    dof_bins = {}
    dof_bins = {}
    dof_bins['bin_start'] = []
    dof_bins['bin_end'] = []
    dof_bin_range = (uulimit - llimit) / float(number_of_bins)
    s = llimit
    for j in range(number_of_bins):
        dof_bins['bin_start'].append(s)
        dof_bins['bin_end'].append(s + dof_bin_range)
        s += dof_bin_range
    return dof_bins


def get_dof_value(bin_no,dof_bins):
    return (dof_bins['bin_start'][bin_no] + dof_bins['bin_end'][bin_no]) / 2.0

def get_dof_bin(value,dof_bins):
    for i in range(len(dof_bins['bin_start'])):
        if value > dof_bins['bin_start'][i] and value < dof_bins['bin_end'][i]:
            return i

def dof_rotate(dof_channel, rot_angle):
    '''
    Only dof1
    dof2 is unaffected from rotations
    '''
    dof1_channel = dof_channel[:,:,:10]
    dof2_channel = dof_channel[:,:,10:]
    dof_bins = get_dof_bins()
    dof_nos = np.argmax(dof1_channel,axis = -1)

    new_dof1_channel = np.zeros(dof1_channel.shape)

    for i in range(dof_nos.shape[0]):
        for j in range(dof_nos.shape[1]):
            dof_value = get_dof_value(dof_nos[i,j],dof_bins)
            new_dof_value = dof_value + rot_angle
            if new_dof_value > np.pi:
                new_dof_value = (new_dof_value - np.pi) + (-np.pi)
            new_dof_bin = get_dof_bin(new_dof_value,dof_bins)
            new_dof1_channel[i,j,new_dof_bin] = dof1_channel[i,j,dof_nos[i,j]]
    new_dof_channel = np.concatenate([new_dof1_channel,dof2_channel],axis = -1)
    return new_dof_channel

def dof1_udflip(value):
    # De-normalize
    denorm_d1 = (value * (2 * np.pi)) + (-np.pi) # [0,1]
    # denorm_d1 = ((value+1) * (np.pi)) + (-np.pi) # [-1,1]
    
    # Flip
    goal_d1 = -denorm_d1
    
    # Normalize
    d1_val_norm = (goal_d1 - (-np.pi)) / (2*np.pi) # [0,1]
    # d1_val_norm = (2*(goal_d1 - (-np.pi)) / (2*np.pi)) - 1 # [-1,1]

    return d1_val_norm

def dof2_udflip(value):
    # De-normalize
    # denorm_d2 = (value * np.pi) + (-np.pi/2) # [0,1]
    denorm_d2 = ((value+1) * (np.pi/2)) + (-np.pi/2) # [-1, 1]

    # Flip
    goal_d2 = -denorm_d2

    # Normalize
    # d2_val_norm = (goal_d2 - (-np.pi/2)) / (np.pi) # [0,1]
    d2_val_norm = (2 * (goal_d2 - (-np.pi/2)) / (np.pi)) - 1 # [-1, 1]
    return d2_val_norm

def remask_channels(arr):
    # data in [0,1]
    # mask = (arr[:,:,0] == 0) 
    # arr[mask] = 0
    
    # data in [-1,1]
    mask = (arr[:,:,0] == -1) 
    arr[mask] = -1
    return arr

def validate_input_tensor(inp):
    assert np.sum(np.isnan(inp)) == 0
    assert len(inp.shape) == 3 
    assert inp.shape[2] == 7 # should have 7 channels (0-2: env image, 3-7: goal dof values)
    for x in range(inp.shape[2]): # all channels should be normalized
        # assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0 # [0,1]
        assert inp[:,:,x].min() >= -1.0 and inp[:,:,x].max() <= 1.0 # [-1,1]
    return True

def validate_label_tensor(lbl):
    assert np.sum(np.isnan(lbl)) == 0
    assert len(lbl.shape) == 3
    assert lbl.shape[2] == 3 # should have 3 channels (0: critical region gaze, 1-2: dof values)
    for x in range(lbl.shape[2]): # all channels should be normalized
        # assert lbl[:,:,x].min() >= 0 and lbl[:,:,x].max() <= 1.0 # [0,1]
        assert lbl[:,:,x].min() >= -1.0 and lbl[:,:,x].max() <= 1.0 # [-1,1]
    return True

if __name__ == "__main__":
    dataPath = 'data'
    envlist = sorted(os.listdir(dataPath))
    for env in envlist:
        envPath = os.path.join(dataPath, env)
        for category in ['inp', 'lbl']:
            path = os.path.join(envPath, category)
            samplefiles = sorted(os.listdir(path))
            for sample in samplefiles: 
                print("processing sample: {category}/{sample}".format(category=category, sample=sample))

                if sample.endswith('.npy') and category == 'inp': # augment input
                    inpfilePath = os.path.join(path, sample)
                    filename = sample.replace('.npy', '')

                    r1Path = os.path.join(path, filename + 'r1.npy')
                    r2Path = os.path.join(path, filename + 'r2.npy')
                    r3Path = os.path.join(path, filename + 'r3.npy')
                    # f1Path = os.path.join(path, filename + 'f1.npy')

                    # split env image and goal condition tensors
                    inp = np.load(inpfilePath)
                    env_tensor = inp[:,:,:3]
                    goal_tensor = inp[:,:,3:]

                    # augment environment tensor
                    env_tensor_r1 = np.rot90(env_tensor, 1, (0, 1))
                    env_tensor_r2 = np.rot90(env_tensor, 2, (0, 1))
                    env_tensor_r3 = np.rot90(env_tensor, 3, (0, 1))
                    # env_tensor_f1 = np.flipud(env_tensor)

                    # updated goal conditions
                    goal = goal_tensor[0,0]
                    goal_r1 = goal_rot(goal, np.pi/2)
                    goal_tensor_r1 = goal_tensor.copy()
                    goal_tensor_r1[:,:] = goal_r1

                    goal_r2 = goal_rot(goal, np.pi)
                    goal_tensor_r2 = goal_tensor.copy()
                    goal_tensor_r2[:,:] = goal_r2
                    
                    goal_r3 = goal_rot(goal, 3*np.pi/2)
                    goal_tensor_r3 = goal_tensor.copy()
                    goal_tensor_r3[:,:] = goal_r3
                    
                    # goal_f1 = goal_udflip(goal)
                    # goal_tensor_f1 = goal_tensor.copy()
                    # goal_tensor_f1[:,:] = goal_f1

                    inp_tensor_r1 = np.concatenate([env_tensor_r1, goal_tensor_r1], axis=2)
                    inp_tensor_r2 = np.concatenate([env_tensor_r2, goal_tensor_r2], axis=2)
                    inp_tensor_r3 = np.concatenate([env_tensor_r3, goal_tensor_r3], axis=2)
                    # inp_tensor_f1 = np.concatenate([env_tensor_f1, goal_tensor_f1], axis=2)

                    assert validate_input_tensor(inp_tensor_r1)
                    assert validate_input_tensor(inp_tensor_r2)
                    assert validate_input_tensor(inp_tensor_r3)
                    # assert validate_input_tensor(inp_tensor_f1)

                    np.save(r1Path, inp_tensor_r1)
                    np.save(r2Path, inp_tensor_r2)
                    np.save(r3Path, inp_tensor_r3)
                    # np.save(f1Path, inp_tensor_f1)

                elif sample.endswith('.npy') and category == 'lbl': # augment label
                    inpfilePath = os.path.join(path, sample)
                    filename = sample.replace('.npy', '')

                    r1Path = os.path.join(path, filename + 'r1.npy')
                    r2Path = os.path.join(path, filename + 'r2.npy')
                    r3Path = os.path.join(path, filename + 'r3.npy')
                    # f1Path = os.path.join(path, filename + 'f1.npy')
                    # m1Path = os.path.join(path, filename + 'm1.npy')

                    # split gaze channel and dof channels
                    inp = np.load(inpfilePath)
                    gaze_tensor = inp[:,:,0]
                    channels_tensor = inp[:,:,1:]

                    # augment environment tensor
                    gaze_tensor_r1 = np.rot90(gaze_tensor, 1, (0, 1)).reshape((224, 224, 1))
                    gaze_tensor_r2 = np.rot90(gaze_tensor, 2, (0, 1)).reshape((224, 224, 1))
                    gaze_tensor_r3 = np.rot90(gaze_tensor, 3, (0, 1)).reshape((224, 224, 1))
                    # gaze_tensor_f1 = np.flipud(gaze_tensor).reshape((224, 224, 1))

                    # updated dof values
                    # rotation_fn = np.vectorize(dof_rotate)
                    channels_tensor_r1 = channels_tensor.copy()
                    channels_tensor_r1 = np.rot90(dof_rotate(channels_tensor, np.pi/2), 1, (0,1))
                    #[:,:,1] = np.rot90(channels_tensor[:,:,1], 1, (0, 1))

                    channels_tensor_r2 = channels_tensor.copy()
                    channels_tensor_r2 = np.rot90(dof_rotate(channels_tensor, np.pi), 2, (0, 1))
                    # channels_tensor_r2[:,:,1] = np.rot90(channels_tensor[:,:,1], 2, (0, 1))

                    channels_tensor_r3 = channels_tensor.copy()
                    channels_tensor_r3 = np.rot90(dof_rotate(channels_tensor, 3*np.pi/2), 3, (0, 1))
                    # channels_tensor_r3[:,:,1] = np.rot90(channels_tensor[:,:,1], 3, (0, 1))


                    lbl_tensor_r1 = np.concatenate([gaze_tensor_r1, channels_tensor_r1], axis=2)
                    lbl_tensor_r2 = np.concatenate([gaze_tensor_r2, channels_tensor_r2], axis=2)
                    lbl_tensor_r3 = np.concatenate([gaze_tensor_r3, channels_tensor_r3], axis=2)
                    # lbl_tensor_f1 = np.concatenate([gaze_tensor_f1, channels_tensor_f1], axis=2)

                    # reapply gaze on channels

                    np.save(r1Path, lbl_tensor_r1)
                    np.save(r2Path, lbl_tensor_r2)
                    np.save(r3Path, lbl_tensor_r3)
                    # np.save(f1Path, lbl_tensor_f1)