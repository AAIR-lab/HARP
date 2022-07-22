from __future__ import division
import pdb
import time
import os, sys
import shutil
import PIL.Image
from scipy.misc import imsave
from scipy.spatial import distance
import openravepy
if not __openravepy_build_doc__:
    from openravepy import *
import numpy as np # from numpy import *

saliencypath = os.path.join('saliency-map-master', 'src')
sys.path.append(saliencypath)
from saliency_map import SaliencyMap
from utils import OpencvIo
import pickle
import cv2
import glob
# import seaborn as sns
import matplotlib.pylab as plt

'''
Script to view how critical regions change based on different threshold values for saliency mapping
'''



class LabelGenerator():

    def __init__(self, robot, envnum, bounds, limits, runnum, labelsize=224):
        self.env = robot.GetEnv()
        self.robot = robot
        self.bounds = bounds
        self.pixelwidth = (bounds[1][0] - bounds[0][0]) / labelsize
        self.pixelBounds = np.fromfunction(lambda i,j: self.setPixelLocations((i,j)), (labelsize,labelsize), dtype=int)
        self.limits = limits
        self.labelsize = labelsize
        self.cdmodel = databases.convexdecomposition.ConvexDecompositionModel(self.robot)
        self.envnum = envnum
        self.runnum = runnum
        self.OMPL = True

        if not self.cdmodel.load():
            self.cdmodel.autogenerate()

        self.basemanip = interfaces.BaseManipulation(self.robot)
        self.setupRobot()

    def setPixelLocations(self, pixel):
        pixminx = self.bounds[0][0] + (pixel[1] * self.pixelwidth)
        pixminy = self.bounds[1][1] - ((pixel[0] + 1) * self.pixelwidth)
        pixmaxx = self.bounds[0][0] + ((pixel[1] + 1) * self.pixelwidth)
        pixmaxy = self.bounds[1][1] - (pixel[0] * self.pixelwidth)
        
        return [(pixminx, pixminy), (pixmaxx, pixmaxy)]

    def pointInBounds(self, point, b):
        '''
        see if a path point is within the pixel bounds; return True if so, else False
        '''
        return ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1])) or \
            ((b[0][0] <= point[0]-0.015 <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1])) or \
            ((b[0][0] <= point[0]+0.015 <= b[1][0]) and (b[0][1] <= point[1] <= b[1][1])) or \
            ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1]-0.015 <= b[1][1])) or \
            ((b[0][0] <= point[0] <= b[1][0]) and (b[0][1] <= point[1]+0.015 <= b[1][1]))

    def pixelInPath(self, pindex, fullpath, xyfullpath, n_channels):
        '''
        calculate pixel value by comparing how many paths cross through the pixel
        '''
        pixminx = self.pixelBounds[0][0][pindex[0]][pindex[1]]
        pixminy = self.pixelBounds[0][1][pindex[0]][pindex[1]]
        pixmaxx = self.pixelBounds[1][0][pindex[0]][pindex[1]]
        pixmaxy = self.pixelBounds[1][1][pindex[0]][pindex[1]]
        
        aveX = (pixmaxx + pixminx) / 2.0
        aveY = (pixmaxy + pixminy) / 2.0
        b = [(pixminx, pixminy), (pixmaxx, pixmaxy)]

        #create dof bins
        n_bin = 8

        d1_bin_start = []
        d1_bin_end = []
        d1_bin_range = 2*np.pi/n_bin
        s = -np.pi
        for i in range(n_bin):
            d1_bin_start.append(s)
            d1_bin_end.append(s + d1_bin_range)
            s += d1_bin_range

        d2_bin_start = []
        d2_bin_end = []
        d2_bin_range = np.pi/n_bin
        s = -np.pi/2
        for i in range(n_bin):
            d2_bin_start.append(s)
            d2_bin_end.append(s + d2_bin_range)
            s += d2_bin_range

        count = 0
        dof_count = np.zeros((2, n_bin))
        for i, path in enumerate(fullpath):
            # xypath = [point[:2] for point in path]
            xypath = xyfullpath[i]
            closestindex = distance.cdist([(aveX,aveY)], xypath).argmin()
            point = path[closestindex]
            if self.pointInBounds(point, b):              
                count += 1 # x, y location counts (1st channel)           
                # rotation(theta) bins counts (dof1 channel)
                for j in range(n_bin):
                    if point[2] >= d1_bin_start[j] and point[2] <= d1_bin_end[j]:
                        dof_count[0][j] += 1              
                # alpha bins counts (dof2 channel)
                for j in range(n_bin):
                    if point[3] >= d2_bin_start[j] and point[3] <= d2_bin_end[j]:
                        dof_count[1][j] += 1
        
        #get bins with highest paths
        max_bins = np.argmax(dof_count, axis=1)
        d1_val = (d1_bin_start[max_bins[0]] + d1_bin_end[max_bins[0]]) / 2
        d2_val = (d2_bin_start[max_bins[1]] + d2_bin_end[max_bins[1]]) / 2

        # normalize to [0,1]
        d1_val_norm = (d1_val - (-np.pi)) / (2*np.pi)
        d2_val_norm = (d2_val - (-np.pi/2)) / (np.pi)

        pvalue = (count/len(fullpath))
        return (pvalue, d1_val_norm, d2_val_norm) #(pvalue, pvalue, pvalue)

    def buildChannels(self, fullpath): 
        '''
        generate channel values for the labels
        '''

        channels = np.zeros((self.labelsize,self.labelsize,3), dtype=float)
        xyfullpath = []
        for path in fullpath:
            xypath = [point[:2] for point in path]
            xyfullpath.append(xypath)

        for row in xrange(channels.shape[0]):
            for col in xrange(channels.shape[1]):
                channels[row][col] = self.pixelInPath((row, col), fullpath, xyfullpath, channels.shape[2])

        npypath = os.path.join('data', 'env' + str(self.envnum), 'data', str(self.envnum) + '.' + str(self.runnum) + '_chnls.npy')
        np.save(npypath, channels)
        return channels
 
    def pixelInObstacle(self, pindex):
        '''
        first obtain pixel bounds in terms of world coordinates
        '''
        pixminx = self.pixelBounds[0][0][pindex[0]][pindex[1]]
        pixminy = self.pixelBounds[0][1][pindex[0]][pindex[1]]
        pixmaxx = self.pixelBounds[1][0][pindex[0]][pindex[1]]
        pixmaxy = self.pixelBounds[1][1][pindex[0]][pindex[1]]

        '''
        see if moving pixel-sized cube to centroid of pixel bound causes collision; return True if so, else False
        '''
        aveX = (pixmaxx + pixminx) / 2.0
        aveY = (pixmaxy + pixminy) / 2.0
        
        collision = False

        body = openravepy.RaveCreateKinBody(self.env, '')
        body.SetName('tracer') # make height of env or robot
        body.InitFromBoxes(np.array([[aveX, aveY, 1.05, self.pixelwidth, self.pixelwidth, 1.0]]), True) # make sure z-coordinate is bounded by shortest wall height
        self.env.AddKinBody(body)
        self.env.UpdatePublishedBodies()

        if self.env.CheckCollision(body): 
            self.env.Remove(body)
            collision = True
        else:
            self.env.Remove(body)

        return collision		    	

    def buildInput(self, inppath, goal_loc):
        '''
        build environment images manually through prodding robot environment
        removes robot
        '''
        oldCC = self.env.GetCollisionChecker()
        total_dofs = self.robot.GetActiveDOF()
        self.env.Remove(self.robot)

        collisionChecker = RaveCreateCollisionChecker(self.env, 'CacheChecker') # bullet CacheChecker pqp
        self.env.SetCollisionChecker(collisionChecker)
        
        # binary env image 0->obstacle 1->free space
        env_tensor = np.zeros((self.labelsize,self.labelsize,3), dtype=int)
        for row in xrange(self.labelsize):
            for col in xrange(self.labelsize):
                if self.pixelInObstacle((row, col)):
                    env_tensor[row][col] = (0, 0, 0)
                else:
                    env_tensor[row][col] = (1, 1, 1)
        
        # goal tensor [0, 1] normalized
        goal_tensor = np.ones((self.labelsize, self.labelsize, total_dofs)) # total dofs
        dof_llimits = self.robot.GetActiveDOFLimits()[0]
        dof_ulimits = self.robot.GetActiveDOFLimits()[1]
        for i, dof_value in enumerate(goal_loc):
            goal_tensor[:,:,i] = (dof_value - dof_llimits[i]) / (dof_ulimits[i] - dof_llimits[i])
        
        #input tensor
        channels = [env_tensor, goal_tensor]
        inp = np.concatenate(channels, axis=2)

        # imsave(imgpath, env_tensor, 'PNG')
        np.save(inppath, inp)
        self.env.SetCollisionChecker(oldCC)
        return env_tensor, goal_tensor

    def buildSM(self, channel_tensor, env_tensor, threshold): #, gazepath, lblpath, smpath):
        '''
        build saliency map and threshold it from motion plots of x,y in 1st channel
        '''
        single_channel = channel_tensor.reshape((self.labelsize,self.labelsize,1))*255
        rgb_image = np.concatenate([single_channel, single_channel, single_channel], axis=2)

        sm = SaliencyMap(rgb_image.astype('uint8'))
        gaze_image = (sm.map - sm.map.min()) / (sm.map.max() - sm.map.min()) # should be 3d array
        gaze_image = gaze_image * 255

        threshold_mask = np.ma.masked_where((env_tensor[:,:,0] == 0) | (gaze_image <= threshold), gaze_image)
        gaze_image[threshold_mask.mask] = 0
        gaze_image[~threshold_mask.mask] = 1

        return sm.map, gaze_image

    def buildLabel(self, channels_tensor, gaze_tensor):
        gaze_tensor = gaze_tensor.reshape((self.labelsize, self.labelsize, 1))
        lbl = np.concatenate([gaze_tensor, channels_tensor[:,:,1:]], axis=2)
        assert lbl.shape[2] == 3

        # consider orientation of only those pixels that are identified as critical points by x,y (1st channel)
        mask = (lbl[:,:,0] == 0) 
        lbl[mask] = 0

        lblpath = os.path.join('data', 'env' + str(self.envnum), 'lbl', str(self.envnum) + '.' + str(self.runnum) + '.npy')
        np.save(lblpath, lbl)

    def setupRobot(self):
        if self.robot.GetName() == 'robot1':
            self.robot.SetDOFVelocities([0.5,0.5,0.5])
            self.robot.SetActiveDOFs([0,1,2])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
        elif self.robot.GetName() == 'lshaped':
            self.robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
            self.robot.SetActiveDOFs([0,1,2,4])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])

    def process_trajectories(self, trajectory_info):
        fullpath = []
        for i, trajectory in enumerate(trajectory_info):

            traj = RaveCreateTrajectory(self.env, '')
            Trajectory.deserialize(traj, trajectory['traj'])

            totaldof = len(self.robot.GetActiveDOFValues())
            movepath = []
            sleeptime = 0.01
            starttime = time.time()
            while time.time()-starttime <= traj.GetDuration():
                curtime = time.time() - starttime
                trajdata = traj.Sample(curtime) 
                movepath.append(trajdata[0:totaldof])
                time.sleep(sleeptime)
            fullpath.append(movepath)

            # for p in movepath:
            #     self.robot.SetActiveDOFValues(p[:totaldof])

        return fullpath

    def generate(self, thresholds):
        '''
        reads all trajectory files in data/env??/data/??,_traj.pkl
        and generates corresponding label and input tensors
        '''
        results = {}
        for threshold in thresholds:
            print('processing {envnum}.{runnum}'.format(envnum=str(self.envnum), runnum=str(self.runnum)))
            trajectories_info = pickle.load(open('data/env{envnum}/data/{envnum}.{runnum}_traj.pkl'.format(envnum=self.envnum, runnum=self.runnum), 'rb'))

            # fullpath = self.process_trajectories(trajectories_info)
            goal_loc = trajectories_info[0]['goal']

            print('creating channels')
            labelstarttime = time.time()
            # channels_tensor = self.buildChannels(fullpath)
            npypath = os.path.join('data', 'env' + str(self.envnum), 'data', str(self.envnum) + '.' + str(self.runnum) + '_chnls.npy')
            channels_tensor = np.load(npypath)
            print('channel creation took: ' + str(time.time() - labelstarttime))
            
            results['path'] = channels_tensor[:,:,0]

            # create env scans
            print('creating input image')
            inppath = os.path.join('data', 'env' + str(self.envnum), 'inp', str(self.envnum) + '.' + str(self.runnum) + '.npy')
            inpstarttime = time.time()
            # env_tensor, goal_tensor = self.buildInput(inppath, goal_loc)
            inp_tensor = np.load(inppath)
            env_tensor = inp_tensor[:,:,:3]
            print('input creation took: ' + str(time.time()-inpstarttime))

            print('creating saliency maps and gaze')
            gazestarttime = time.time()
            sm, gaze_tensor = self.buildSM(channels_tensor[:,:,0], env_tensor, threshold)
            print('gaze took: '+str(time.time()-gazestarttime))

            results['sm'] = sm
            results[threshold] = gaze_tensor

            # print('combining gaze with channels')
            # mergestarttime = time.time()
            # self.buildLabel(channels_tensor, gaze_tensor)
            # print('merging took: ' + str(time.time() - mergestarttime))
        return results

if __name__ == '__main__':

    # envPath = argv[1]
	# envnum = float(argv[2])
	# runnum = int(argv[3])
    start = 1
    while True:
        f, axarr = plt.subplots(10, 5)

        for i, j in enumerate(range(start, start+10)):
            envPath = 'envs3d/21.0.xml'
            envnum = 21.0
            runnum = j #39 #np.random.randint(1, 101)

            env = Environment()
            env.Load(envPath)
            
            # env.SetViewer('qtcoin')
            
            # set collision checker to Bullet (default collision checker might not recognize cylinder collision for Ubuntu) (causes issues when moving joints)
            collisionChecker = RaveCreateCollisionChecker(env, 'fcl_')
            env.SetCollisionChecker(collisionChecker)
            
            robot = env.GetRobots()[0]

            bounds = np.array([[-2.5, -2.5, -3.14], [2.5, 2.5, 3.14]]) # all random envs are built using the same base world, thus they have the same world bounds
            envmin = np.array([-2.5, -2.5, -2.5])
            envmax = np.array([2.5, 2.5, 2.5])

            generator = LabelGenerator(robot, envnum, bounds, [envmin, envmax], runnum)
            thresholds = [100, 70, 50]
            results = generator.generate()
            axarr[i][0].imshow(results['path'])
            axarr[i][1].imshow(results['sm'])
            axarr[i][2].imshow(results[thresholds[0]])
            axarr[i][3].imshow(results[thresholds[1]])
            axarr[i][4].imshow(results[thresholds[2]])
        plt.show()
        start += 10