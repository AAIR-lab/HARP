import os
import numpy as np
import openravepy
from scipy.spatial.transform import Rotation

'''
Processes generated data samples to create 4 new samples:
Rotate 90, 180, 270 
Output new labels appended with suffixes r1, r2, r3

Augments inputs, labels remain same for 3d-8dof
'''

def validate_input_tensor(inp):
    assert np.sum(np.isnan(inp)) == 0
    assert len(inp.shape) == 4
    assert inp.shape[-1] == 17 # should have 17 channels (0-2: env image, 3-10: goal dof, 11-16: fetch location)
    for x in range(inp.shape[-1]): # all channels should be normalized
        assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0 # [0,1]
        # assert inp[:,:,x].min() >= -1.0 and inp[:,:,x].max() <= 1.0 # [-1,1]
    return True

def rotate_env(env_tensor, theta):
    '''
    perform rotation about center of tensor
    Env is made up of 0(obstacle) and 1(free space) values for voxels.
    Only need to move 0 valued voxels to their new location
    '''

    indices = np.where(env_tensor[:,:,:,0] == 0)
    
    rotated_env = np.ones((env_tensor.shape[0], env_tensor.shape[1], env_tensor.shape[2]))
    
    for i in range(len(indices[0])):
        coords = [indices[0][i], indices[1][i], indices[2][i]]

        # shift origin to center of tensor (z remains same)
        coords[0] += -(labelsize-1)/2.
        coords[1] += -(labelsize-1)/2.

        axis = [0, 0, 1]
        axis = axis / np.linalg.norm(axis)  # normalize the rotation vector first
        
        rot = Rotation.from_rotvec(theta * axis)
        new_coords = rot.apply(coords)

        #shift origin back
        new_coords[0] += (labelsize-1)/2.
        new_coords[1] += (labelsize-1)/2.
        new_coords = np.round(new_coords).astype(int)
        rotated_env[new_coords[0], new_coords[1], new_coords[2]] = 0
    
    rotated_env = np.stack([rotated_env, rotated_env, rotated_env], axis=3)
    return rotated_env

def fetch_loc_rotate(fetch_loc_tensor, theta):

    fetch_loc = fetch_loc_tensor[0,0,0,:]
    fetch_xyz = fetch_loc[0:3]
    fetch_axisAngles = fetch_loc[3:]
    
    denorm_xyz = []
    for i in range(3):
        denorm_xyz.append((fetch_xyz[i] * (bounds[1][i] - bounds[0][i])) + (bounds[0][i]))

    denorm_axisAngles = []
    for i in range(3):
        denorm_axisAngles.append((fetch_axisAngles[i] * (2*np.pi)) + (-np.pi))
    
    # Rotate xyz
    axis = [0, 0, 1]
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(theta * axis)
    new_xyz = rot.apply(denorm_xyz)

    # Normalize xyz
    norm_xyz = []
    for i in range(3):
        norm_xyz.append((new_xyz[i] - (bounds[0][i])) / (bounds[1][i] - bounds[0][i]))

    # Rotate angles
    new_axisAngles = denorm_axisAngles
    new_axisAngles[2] += theta
    if new_axisAngles[2] > np.pi:
        new_axisAngles[2] = (new_axisAngles[2] - np.pi) + (-np.pi)
    
    # Normalize
    norm_axisAngles = []
    for i in range(3):
        norm_axisAngles.append((new_axisAngles[i] - (-np.pi)) / (2*np.pi))

    updated_fetch_loc = norm_xyz + norm_axisAngles
    updated_fetch_loc_tensor = fetch_loc_tensor.copy()
    for i in range(6):
        updated_fetch_loc_tensor[:,:,:,i] = updated_fetch_loc[i]

    return updated_fetch_loc_tensor


if __name__ == "__main__":
    bounds = [[-2.5, -2.5, -0.001], [2.5, 2.5, 2.5]]
    labelsize = 64
    dataPath = 'data'
    envlist = sorted(os.listdir(dataPath)) # ['env1.0']
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
                    f1Path = os.path.join(path, filename + 'f1.npy')

                    # split env image and goal condition tensors
                    inp = np.load(inpfilePath)
                    env_tensor = inp[:,:,:,:3].copy()
                    goal_tensor = inp[:,:,:,3:11].copy()
                    fetch_loc_tensor = inp[:,:,:,11:].copy()

                    # augment environment tensor
                    env_tensor_r1 = rotate_env(env_tensor, np.pi/2)
                    env_tensor_r2 = rotate_env(env_tensor, np.pi)
                    env_tensor_r3 = rotate_env(env_tensor, 3*np.pi/2)

                    # update fetch location
                    fetch_loc_tensor_r1 = fetch_loc_rotate(fetch_loc_tensor, np.pi/2)
                    fetch_loc_tensor_r2 = fetch_loc_rotate(fetch_loc_tensor, np.pi)
                    fetch_loc_tensor_r3 = fetch_loc_rotate(fetch_loc_tensor, 3*np.pi/2)

                    inp_tensor_r1 = np.concatenate([env_tensor_r1, goal_tensor, fetch_loc_tensor_r1], axis=3)
                    inp_tensor_r2 = np.concatenate([env_tensor_r2, goal_tensor, fetch_loc_tensor_r2], axis=3)
                    inp_tensor_r3 = np.concatenate([env_tensor_r3, goal_tensor, fetch_loc_tensor_r3], axis=3)

                    assert validate_input_tensor(inp_tensor_r1)
                    assert validate_input_tensor(inp_tensor_r2)
                    assert validate_input_tensor(inp_tensor_r3)

                    np.save(r1Path, inp_tensor_r1)
                    np.save(r2Path, inp_tensor_r2)
                    np.save(r3Path, inp_tensor_r3)

                elif sample.endswith('.npy') and category == 'lbl': # augment label
                    # if labels are distributions then they don't need to be augmented

                    lblfilePath = os.path.join(path, sample)
                    filename = sample.replace('.npy', '')

                    r1Path = os.path.join(path, filename + 'r1.npy')
                    r2Path = os.path.join(path, filename + 'r2.npy')
                    r3Path = os.path.join(path, filename + 'r3.npy')

                    lbl = np.load(lblfilePath)

                    np.save(r1Path, lbl)
                    np.save(r2Path, lbl)
                    np.save(r3Path, lbl)
