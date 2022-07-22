import time
import os, sys
import shutil
import PIL.Image
from scipy.misc import imsave
from scipy.spatial import distance
import openravepy
from openravepy import *
import numpy as np

# saliencypath = os.path.join('saliency-map-master', 'src')
# sys.path.append(saliencypath)
# from saliency_map import SaliencyMap
import cv2



sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
import fetch

import pickle
import cv2
import glob
import matplotlib.pylab as plt
import multiprocessing as mp


'''
generates inputs and (N x N x N x CHANNELS) shaped tensors as labels
'''

class LabelGenerator():

    def __init__(self, env, envnum, runnum, bounds, labelsize=224):        
        self.env = env
        self.robot = env.GetRobots()[0]
        self.bounds = bounds
        self.labelsize = labelsize
        self.pixeldims = [0,0,0]
        self.pixeldims[0] = (bounds[1][0] - bounds[0][0]) / self.labelsize # x
        self.pixeldims[1] = (bounds[1][1] - bounds[0][1]) / self.labelsize # y
        self.pixeldims[2] = (bounds[1][2] - bounds[0][2]) / self.labelsize # z
        self.pixelBounds = np.fromfunction(lambda i, j, k: self.setPixelLocations((i, j, k)), (self.labelsize,self.labelsize, self.labelsize), dtype=int)
        self.envnum = envnum
        self.runnum = runnum
        self.labelchannels = 9
        self.number_of_bins = 10
        self.number_of_dof_bins = 10
        self.number_of_dofs = self.robot.GetActiveDOF()
        self.fullpath = None
        self.xyzfullpath = None
        self.goal_loc = None
        self.setupRobot()
        self.load_motionplans()

    def setPixelLocations(self, pixel):
        pixminx = self.bounds[0][0] + (pixel[0] * self.pixeldims[0])
        pixmaxx = self.bounds[0][0] + ((pixel[0] + 1) * self.pixeldims[0])
        
        pixminy = self.bounds[0][1] + (pixel[1] * self.pixeldims[1])
        pixmaxy = self.bounds[0][1] + ((pixel[1] + 1) * self.pixeldims[1])
        
        pixminz = self.bounds[0][2] + (pixel[2] * self.pixeldims[2])
        pixmaxz = self.bounds[0][2] + ((pixel[2] + 1) * self.pixeldims[2])

        return [(pixminx, pixminy, pixminz), (pixmaxx, pixmaxy, pixmaxz)]

    def pointInBounds(self, point, b):
        '''
        see if a path point is within the pixel bounds; return True if so, else False
        '''
        return ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1]) and b[0][2] <= point[2] <= b[1][2]) or \
            ((b[0][0] <= point[0]-0.015 <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1]) and b[0][2] <= point[2] <= b[1][2]) or \
            ((b[0][0] <= point[0]+0.015 <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1]) and b[0][2] <= point[2] <= b[1][2]) or \
            ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1]-0.015 <= b[1][1]) and b[0][2] <= point[2] <= b[1][2]) or \
            ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1]+0.015 <= b[1][1]) and b[0][2] <= point[2] <= b[1][2]) or \
            ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1]) and b[0][2] <= point[2]-0.015 <= b[1][2]) or \
            ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1]) and b[0][2] <= point[2]+0.015 <= b[1][2])

 
    def pixelInObstacle(self, pindex):
        '''
        see if moving pixel-sized cube to centroid of pixel bound causes collision; return True if so, else False
        '''
        pixminx = self.pixelBounds[0][0][pindex[0]][pindex[1]][pindex[2]]
        pixminy = self.pixelBounds[0][1][pindex[0]][pindex[1]][pindex[2]]
        pixminz = self.pixelBounds[0][2][pindex[0]][pindex[1]][pindex[2]]
        pixmaxx = self.pixelBounds[1][0][pindex[0]][pindex[1]][pindex[2]]
        pixmaxy = self.pixelBounds[1][1][pindex[0]][pindex[1]][pindex[2]]
        pixmaxz = self.pixelBounds[1][2][pindex[0]][pindex[1]][pindex[2]]
        
        aveX = (pixmaxx + pixminx) / 2.0
        aveY = (pixmaxy + pixminy) / 2.0
        aveZ = (pixmaxz + pixminz) / 2.0
        
        collision = False

        body = openravepy.RaveCreateKinBody(self.env, '')
        body.SetName('tracer')
        body.InitFromBoxes(np.array([[aveX, aveY, aveZ, self.pixeldims[0], self.pixeldims[1], self.pixeldims[2]]]), True)
        self.env.AddKinBody(body)
        self.env.UpdatePublishedBodies()

        if self.env.CheckCollision(body): 
            self.env.Remove(body)
            collision = True
        else:
            self.env.Remove(body)

        return collision		    	

    def buildInput(self):
        '''
        build environment images manually through prodding robot environment
        removes robot
        '''
        env_tensor = None
        sample_file = os.path.join(INPDIR, str(self.envnum) + '.1.npy')
        total_dofs = self.robot.GetActiveDOF()
        if os.path.isfile(sample_file):
            sample_data = np.load(sample_file)
            env_tensor = sample_data[:,:,:,:3]
        else:
            oldCC = self.env.GetCollisionChecker()
           
            self.env.Remove(self.robot)

            collisionChecker = RaveCreateCollisionChecker(self.env, 'CacheChecker') # bullet CacheChecker pqp
            self.env.SetCollisionChecker(collisionChecker)
            
            # binary env image 0->obstacle 1->free space
            env_tensor = np.zeros((self.labelsize,self.labelsize, self.labelsize, 3), dtype=int)
            for i in xrange(self.labelsize):
                for j in xrange(self.labelsize):
                    for k in xrange(self.labelsize):
                        if self.pixelInObstacle((i, j, k)):
                            env_tensor[i][j][k] = (0, 0, 0) # [0,1]
                            # env_tensor[i][j][k] = (-1,-1,-1) # [-1,1] 
                        else:
                            env_tensor[i][j][k] = (1, 1, 1)
            
            self.env.SetCollisionChecker(oldCC)
        
        # goal tensor normalized
        goal_tensor = np.ones((self.labelsize, self.labelsize, self.labelsize, total_dofs)) # total dofs
        dof_llimits = self.robot.GetActiveDOFLimits()[0]
        dof_ulimits = self.robot.GetActiveDOFLimits()[1]
        for i, dof_value in enumerate(self.goal_loc):
            goal_tensor[:,:,:,i] = (dof_value - dof_llimits[i]) / (dof_ulimits[i] - dof_llimits[i]) # [0,1]
            # goal_tensor[:,:,i] = (2 * ((dof_value - dof_llimits[i]) / (dof_ulimits[i] - dof_llimits[i]))) - 1 # [-1,1]
        
        #input tensor
        channels = [env_tensor, goal_tensor]
        inp = np.concatenate(channels, axis=3)

        # validate
        assert np.sum(np.isnan(inp)) == 0
        assert len(inp.shape) == 4
        assert inp.shape[0] == self.labelsize and inp.shape[1] == self.labelsize and inp.shape[2] == self.labelsize
        assert inp.shape[3] == 11 # should have 7 channels (0-2: env image, 3-10: goal dof values)
        for x in range(inp.shape[3]): # all channels should be normalized
            assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0
        
        inppath = os.path.join(INPDIR, '{envnum}.{runnum}.npy'.format(envnum=self.envnum, runnum=self.runnum))
        np.save(inppath, inp)
        return env_tensor, goal_tensor
    
    def buildSM3D(self, channel_tensor, env_tensor, threshold):
        
        # Threshold CR
        gaze_tensor = channel_tensor.copy()
        gaze_tensor = gaze_tensor.reshape((self.labelsize, self.labelsize, self.labelsize, 1))
        env_tensor = env_tensor.reshape((self.labelsize, self.labelsize, self.labelsize, 1))

        threshold_mask = np.ma.masked_where((env_tensor == 0) | (gaze_tensor < threshold), gaze_tensor)
        gaze_tensor[threshold_mask.mask] = 0
        gaze_tensor[~threshold_mask.mask] = 1

        return gaze_tensor

    def buildLabel(self):
        n_bin = self.number_of_dof_bins
        llimits = self.robot.GetActiveDOFLimits()[0]
        ulimits = self.robot.GetActiveDOFLimits()[1]
        dof_ranges = ulimits - llimits
        dof_bins = {}
        xyz_bins = {}
        for i in range(3):
            step = (self.bounds[1][i] - self.bounds[0][i]) / float(self.labelsize)
            xyz_bins[i] = {}
            xyz_bins[i]['bin_start'] = []
            xyz_bins[i]['bin_end'] = []
            s = self.bounds[0][i]
            for j in range(self.labelsize):
                xyz_bins[i]['bin_start'].append(s)
                xyz_bins[i]['bin_end'].append(s+step)
                s += step
        
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

        goal_bins = []
        for i in range(len(self.goal_loc)):
            for j in range(self.number_of_dof_bins):
                if self.goal_loc[i] >= dof_bins[i]['bin_start'][j] and self.goal_loc[i] < dof_bins[i]['bin_end'][j]:
                    goal_bins.append(j)

        xyz_distribution = np.zeros((self.labelsize,self.labelsize,self.labelsize))

        dof_distributions = np.zeros((len(dof_ranges),self.labelsize,self.labelsize,self.labelsize,self.number_of_dof_bins))

        for i in range(len(self.fullpath)):
            xyz_flag = np.zeros(xyz_distribution.shape)
            dof_flag = np.zeros(dof_distributions.shape)

            path = self.fullpath[i]
            xyzpath = self.xyzfullpath[i]

            for j in range(len(path)):
                point = path[j]
                xyzpoint = xyzpath[j]
                x_bin = self.get_bin(xyz_bins,xyzpoint[0],0)
                y_bin = self.get_bin(xyz_bins,xyzpoint[1],1)
                z_bin = self.get_bin(xyz_bins,xyzpoint[2],2)

                if xyz_flag[x_bin,y_bin,z_bin] == 0:
                    xyz_distribution[x_bin,y_bin,z_bin] += 1.0
                    xyz_flag[x_bin,y_bin,z_bin] = 1 
                
                for i in range(len(dof_ranges)):
                    dof_i_bin = self.get_bin(dof_bins,point[i],i)
                    if dof_flag[i,x_bin,y_bin,z_bin,dof_i_bin] == 0:
                        dof_distributions[i,x_bin,y_bin,z_bin,dof_i_bin] += 1.0
                        dof_flag[i,x_bin,y_bin,z_bin,dof_i_bin] = 1
        
        dof_channels = np.zeros((self.labelsize,self.labelsize,self.labelsize,len(dof_ranges)))


        for i in range(dof_distributions.shape[0]):
            dof_i_distribution = dof_distributions[i,:,:,:,:]
            dof_argmax = np.argmax(dof_i_distribution,axis = -1)
            dof_channels[:,:,:,i] = dof_argmax

        pd_xyz = xyz_distribution / float(len(self.fullpath))
        pd_xyz = pd_xyz.reshape((pd_xyz.shape[0],pd_xyz.shape[1],pd_xyz.shape[2],1))

        pd = np.concatenate([pd_xyz,dof_channels],axis = -1)

        pd = np.nan_to_num(pd)

        assert pd[:,:,:,0].min() >= 0.0 and pd[:,:,:,0].max() <= 1.0
        assert pd[:,:,:,1:].min() >= 0.0 and pd[:,:,:,1:].max() <= self.number_of_dof_bins
        assert len(pd.shape) == 4
        assert pd.shape[0] == self.labelsize and pd.shape[1] == self.labelsize and pd.shape[2] == self.labelsize and pd.shape[3] == self.number_of_dofs + 1

        return pd

    def get_bin(self,dof_bins,value,dof_number):
        for j in range(len(dof_bins[dof_number]["bin_start"])):
            if value >= dof_bins[dof_number]["bin_start"][j] and value < dof_bins[dof_number]["bin_end"][j]:
                return j
        return -1

    def setupRobot(self):
        if self.robot.GetName() == 'robot1':
            self.robot.SetDOFVelocities([0.5,0.5,0.5])
            self.robot.SetActiveDOFs([0,1,2])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
        elif self.robot.GetName() == 'lshaped':
            self.robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
            self.robot.SetActiveDOFs([0,1,2,4])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])
        elif self.robot.GetName() == 'fetch':
            llimits = self.robot.GetDOFLimits()[0]
            ulimits = self.robot.GetDOFLimits()[1]
            # DOFs 2, 12, 14 are circular and need their limits to be set to [-3.14, 3.14] as stated by planner
            llimits[2] = llimits[12] = llimits[14] = -np.pi
            ulimits[2] = ulimits[12] = ulimits[14] = np.pi
            self.robot.SetDOFLimits(llimits, ulimits)

    def load_motionplans(self):
        print('Loading motion plans..')
        trajectory_info = pickle.load(open(DATADIR + '/{envnum}.{runnum}_traj.pkl'.format(envnum=self.envnum, runnum=self.runnum), 'rb'))
        self.goal_loc = trajectory_info[0]['goal']
        fullpath = []
        for trajectory in trajectory_info:
            fullpath.append(trajectory['path'])
        self.fullpath = fullpath
        self.xyzfullpath = []
        for path in fullpath:
            xyzpath = []
            for point in path:
                self.robot.SetActiveDOFValues(point)
                end_effector_transform = self.robot.GetManipulators()[0].GetTransform()
                xyzpath.append(end_effector_transform[:3,3])
            self.xyzfullpath.append(xyzpath)

    def custom_sm(self,pd,threshold):
        mean = np.mean(pd)
        std = np.std(pd)
        mask = np.ma.masked_where( pd == 1.0 ,pd)
        pd_copy = pd.copy()
        pd_copy[mask.mask] = 0.0
        m = np.max(pd_copy)
        new_pd = pd/m
        mask = np.ma.masked_where(new_pd < threshold, new_pd)
        new_pd[mask.mask] = 0
        new_pd[~mask.mask] = 1.0
        # new_pd = self.smoothen_sm(new_pd)
        # mask = np.ma.masked_where(new_pd < m,new_pd)
        # new_pd[mask.mask] = 0
        # new_pd[~mask.mask] = 1.0
        return new_pd


    def smoothen_sm(self,pd,n=2):
        return cv2.GaussianBlur(pd,(5,5),0)
        
    def mergeLabel(self,lbl,gaze):
        gaze = gaze.reshape((self.labelsize,self.labelsize,self.labelsize,1))
        lbl = np.concatenate([gaze,lbl[:,:,:,1:]],axis = -1)


        mask = np.ma.masked_where(lbl[:,:,:] == 0, lbl)
        lbl[mask.mask] = 0

        print "gaze max: ", np.max(gaze)
        print "lbl max: ", np.max(lbl[:,:,:,1:])

        assert np.sum(np.isnan(lbl)) == 0
        assert len(lbl.shape) == 4
        assert lbl.shape[-1] == 9

        lblpath = os.path.join(LBLDIR, str(self.envnum) + '.' + str(self.runnum) + '.npy')
        np.save(lblpath, lbl)

    def generate(self, threshold):
        '''
        reads all trajectory files in data/env??/data/??_traj.pkl
        and generates corresponding label and input tensors
        '''

        print('creating channels')
        labelstarttime = time.time()
        label_tensor = self.buildLabel()
        # npypath = os.path.join(DATADIR, str(self.envnum) + '.' + str(self.runnum) + '_chnls.npy')
        # channels_tensor = np.load(npypath)
        print('channel creation took: ' + str(time.time() - labelstarttime))

        # create env scans
        print('creating input image')
        inpstarttime = time.time()
        env_tensor, goal_tensor = self.buildInput()
        # inppath = os.path.join(INPDIR, '{envnum}.{runnum}.npy'.format(envnum=self.envnum, runnum=self.runnum))
        # inp = np.load(inppath)
        # env_tensor = inp[:,:,:,:3]
        print('input creation took: ' + str(time.time()-inpstarttime))
        print('creating saliency maps and gaze')
        gazestarttime = time.time()
        gaze_tensor = self.custom_sm(label_tensor[:,:,:,0],threshold)
        print('gaze took: '+str(time.time()-gazestarttime))

        print('combining gaze with channels')
        mergestarttime = time.time()
        self.mergeLabel(label_tensor, gaze_tensor)
        print('merging took: ' + str(time.time() - mergestarttime))


if __name__ == '__main__':

    envPath = sys.argv[1]
    envnum = float(sys.argv[2])
    runnum = int(sys.argv[3])
    threshold = float(sys.argv[4])
    visualize = False

    # envPath = 'envs/3d/1.0.dae'
    # envnum = 1.0
    # runnum = 1
    # threshold = 0.7
    # visualize = False

    env = Environment()
    env.Load(envPath)
    if visualize: 
        env.SetViewer('qtcoin')
    
    # set collision checker to Bullet (default collision checker might not recognize cylinder collision for Ubuntu) (causes issues when moving joints)
    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    collisionChecker.SetCollisionOptions(CollisionOptions.Contacts)
    env.SetCollisionChecker(collisionChecker)
    
    initial_transform = pickle.load(open(os.path.join('envs','3d','fetch_transforms', str(envnum) + '.pkl'), 'rb'))
    fetch_robot = fetch.FetchRobot(env, initial_transform)
    robot = env.GetRobots()[0]
    robot_traform = robot.GetTransform()
    # bounds = [[-1, -1, 0], [1, 1, 2]]
    bounds = [[robot_traform[0,3]-1,robot_traform[1,3]-1,0],[robot_traform[0,3]+1,robot_traform[1,3]+1,2.0]]

    DATADIR = os.path.join('data-v2', 'env' + str(envnum), 'data')
    INPDIR = os.path.join('data-v2', 'env' + str(envnum), 'inp')
    if not os.path.isdir(INPDIR):
        os.makedirs(INPDIR)
    LBLDIR = os.path.join('data-v2', 'env' + str(envnum), 'lbl')
    if not os.path.isdir(LBLDIR):
        os.makedirs(LBLDIR)

    print('*********processing {envnum}.{runnum}*********'.format(envnum=str(envnum), runnum=str(runnum)))
    generator = LabelGenerator(env, envnum, runnum, bounds, 64)
    generator.generate(threshold)
