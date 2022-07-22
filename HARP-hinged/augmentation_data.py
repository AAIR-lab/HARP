import numpy as np
import os
import pickle
from openravepy import *
import math

class Augmenter(object):

    def __init__(self,envnum,envpath,problemfile):
        self.envnum = envnum
        self.envpath = envpath
        self.problemfile = problemfile
        self.trajfile = "data/env{}/data/{}_traj.pkl".format(envnum,problemfile)
        self.data = pickle.load(open(self.trajfile,"rb"))
        self.augment()


    def goal_rot(self, goal, rot_angle):
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
        # new_goal[0] = (updated_loc[0,0] - (-2.5)) / (5.)
        # new_goal[1] = (updated_loc[1,0] - (-2.5)) / (5.)
        # [-1,1]
        new_goal[0] = (2*(updated_loc[0,0] - (-2.5)) / (5.)) - 1
        new_goal[1] = (2*(updated_loc[1,0] - (-2.5)) / (5.)) - 1

        # Denormalize to [-pi, pi]
        # denorm_d1 = (goal[2] * (2 * np.pi)) + (-np.pi) # [0,1]
        denorm_d1 = ((goal[2]+1) * (np.pi)) + (-np.pi) # [-1,1]
        new_goal[2] = denorm_d1 + rot_angle
        if new_goal[2] > np.pi:
            new_goal[2] = (new_goal[2] - np.pi) + (-np.pi)
        # normalize 
        # [0, 1]
        # new_goal[2] = (new_goal[2] - (-np.pi)) / (2*np.pi)
        # [-1,1]
        new_goal[2] = (2*(new_goal[2] - (-np.pi)) / (2*np.pi)) - 1
        
        # Dof 2 is unaffected by rotation. It is relative to the main body
        new_goal[3] = goal[3]
        return new_goal

    def goal_udflip(self,goal):
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
        # d1_val_norm = (goal_d1 - (-np.pi)) / (2*np.pi) # [0,1]
        d1_val_norm = (2*(goal_d1 - (-np.pi)) / (2*np.pi)) - 1 # [-1,1]
        new_goal[2] = d1_val_norm
        
        # DOF 2
        # De-Normalize
        # denorm_d2 = (goal[3] * np.pi) + (-np.pi/2) # [0,1]
        denorm_d2 = ((goal[3]+1) * (np.pi/2)) + (-np.pi/2) # [-1, 1]
        
        # angle is flipped
        goal_d2 = -denorm_d2
        
        # Normalize
        # d2_val_norm = (goal_d2 - (-np.pi/2)) / (np.pi) # [0,1]
        d2_val_norm = (2 * (goal_d2 - (-np.pi/2)) / (np.pi)) - 1 # [-1, 1]
        new_goal[3] = d2_val_norm
        
        return new_goal

    def dof_rotate(self, value, rot_angle):
        '''
        Only dof1
        dof2 is unaffected from rotations
        '''

        # De-Normalize
        # denorm_d1 = (value * (2 * np.pi)) + (-np.pi) # [0,1]
        denorm_d1 = ((value+1) * (np.pi)) + (-np.pi) # [-1,1]

        # Rotate
        new_value = denorm_d1 + rot_angle
        if new_value > np.pi:
            new_value = (new_value - np.pi) + (-np.pi)
        
        # Normalize
        # d1_val_norm = (new_value - (-np.pi)) / (2*np.pi) # [0,1]
        d1_val_norm = (2*(new_value - (-np.pi)) / (2*np.pi)) - 1 # [-1,1]

        return d1_val_norm

    def dof1_udflip(self, value):
        # De-normalize
        # denorm_d1 = (value * (2 * np.pi)) + (-np.pi) # [0,1]
        denorm_d1 = ((value+1) * (np.pi)) + (-np.pi) # [-1,1]
        
        # Flip
        goal_d1 = -denorm_d1
        
        # Normalize
        # d1_val_norm = (goal_d1 - (-np.pi)) / (2*np.pi) # [0,1]
        d1_val_norm = (2*(goal_d1 - (-np.pi)) / (2*np.pi)) - 1 # [-1,1]

        return d1_val_norm

    def dof2_udflip(self, value):
        # De-normalize
        # denorm_d2 = (value * np.pi) + (-np.pi/2) # [0,1]
        denorm_d2 = ((value+1) * (np.pi/2)) + (-np.pi/2) # [-1, 1]

        # Flip
        goal_d2 = -denorm_d2

        # Normalize
        # d2_val_norm = (goal_d2 - (-np.pi/2)) / (np.pi) # [0,1]
        d2_val_norm = (2 * (goal_d2 - (-np.pi/2)) / (np.pi)) - 1 # [-1, 1]
        return d2_val_norm
    
    def augment():
        new_trajs = []
        for traj in self.data:
            goal = traj["goal"]
            points = traj["points"]
            augmented_goal_r90 = self.goal_rot(goal,math.pi/2.0)
            augmented_goal_r180 = self.goal_rot(goal,math.pi)
            augmented_goal_r270 = self.goal_rot(goal,-math.pi/2.0)
            augmented_goal_flip = self.goal_udflip(goal)

            for point in points:
                augmented_goal_r90 = self.
        pass




if __name__ == "__main__":

    envnum = "5.1"
    envpath = "envs3d/{}.xml".format(envnum)
    runnum = "1"
    problemfile = envnum + "." + runnum

    datafile = "data/env{}/{}_traj.pkl".format(envnum,problemfile)
    Augmenter(envnum,envpath,problemfile)