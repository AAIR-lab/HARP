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

# saliencypath = os.path.join('saliency-map-master', 'src')
# sys.path.append(saliencypath)
# from saliency_map import SaliencyMap
# from utils import OpencvIo
import pickle
import cv2
import glob
import matplotlib.pylab as plt

class InputGenerator():

    def __init__(self, robot, envnum, bounds, limits, runnum, labelsize=224):
        self.env = robot.GetEnv()
        self.robot = robot
        self.bounds = bounds
        self.pixelwidth = (bounds[1][0] - bounds[0][0]) / 224
        self.pixelBounds = np.fromfunction(lambda i,j: self.setPixelLocations((i,j)), (224,224), dtype=int)
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
        
        # binary env image -1/0->obstacle 1->free space
        env_tensor = np.zeros((224,224,3), dtype=int)
        for row in xrange(224):
            for col in xrange(224):
                if self.pixelInObstacle((row, col)):
                    env_tensor[row][col] = (0, 0, 0) # [0,1]
                    # env_tensor[row][col] = (-1,-1,-1) # [-1,1] 
                else:
                    env_tensor[row][col] = (1, 1, 1)
        
        # Add border to input image
        # border_pixel_size = 8
        # pixel_value = (0,0,0)
        # env_tensor[:border_pixel_size,:] = pixel_value
        # env_tensor[:,:border_pixel_size] = pixel_value
        # env_tensor[:,-border_pixel_size:] = pixel_value
        # env_tensor[-border_pixel_size:,:] = pixel_value

        # plt.imshow(env_tensor[:,:,0], cmap='gray')
        # plt.show()

        self.env.SetCollisionChecker(oldCC)

        # goal tensor normalized
        goal_tensor = np.ones((224, 224, total_dofs)) # total dofs
        dof_llimits = self.robot.GetActiveDOFLimits()[0]
        dof_ulimits = self.robot.GetActiveDOFLimits()[1]
        for i, dof_value in enumerate(goal_loc):
            goal_tensor[:,:,i] = (dof_value - dof_llimits[i]) / (dof_ulimits[i] - dof_llimits[i]) # [0,1]
            # goal_tensor[:,:,i] = (2 * ((dof_value - dof_llimits[i]) / (dof_ulimits[i] - dof_llimits[i]))) - 1 # [-1,1]
        
        #input tensor
        channels = [env_tensor, goal_tensor]
        inp = np.concatenate(channels, axis=2)

        # validate
        assert np.sum(np.isnan(inp)) == 0
        assert len(inp.shape) == 3 
        assert inp.shape[2] == 7 # should have 7 channels (0-2: env image, 3-7: goal dof values)
        for x in range(inp.shape[2]): # all channels should be normalized
            assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0

        np.save(inppath, inp)
        return env_tensor, goal_tensor

    def setupRobot(self):
        if self.robot.GetName() == 'robot1':
            self.robot.SetDOFVelocities([0.5,0.5,0.5])
            self.robot.SetActiveDOFs([0,1,2])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
        elif self.robot.GetName() == 'lshaped':
            self.robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
            self.robot.SetActiveDOFs([0,1,2,4])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])

    def generate(self, goal_loc):
        '''
        reads all trajectory files in data/env??/data/??,_traj.pkl
        and generates corresponding label and input tensors
        '''

        # create env scans
        print('creating input image')
        inppath = os.path.join('test', "env"+str(self.envnum), str(self.envnum)+"."+str(self.runnum) + '.npy')
        inpstarttime = time.time()
        env_tensor, goal_tensor = self.buildInput(inppath, goal_loc)
        print('input creation took: ' + str(time.time()-inpstarttime))

if __name__ == '__main__':

    # envPath = sys.argv[1]
	# envnum = float(sys.argv[2])
	# runnum = int(sys.argv[3])

    envPath = 'envs3d/8.0.xml'
    envnum = 8.0
    runnum = 1
    goal_loc = [2.,   1.,   1.57, 0.  ] # 8.0.1
    # goal_loc = [-2., -1.,  0.,  0.] # 8.0.2
    # goal_loc = [-2.,  1. , 0.,  0.] # 8.0.3
    # goal_loc = [ 2., -1.,  0.,  0.] # 8.0.4
    # goal_loc = [-0.96970837, -0.82476306, -2.42265842,  0.55285693] # 10.2.1
    # goal_loc = [2.,   1.6,  1.57, 0.  ] # 21.0.1

    f = open("./test/env{}/goal_{}_{}.pkl".format(envnum,envnum,runnum),"wb")
    pickle.dump(goal_loc,f)

    env = Environment()
    env.Load(envPath)
    
    env.SetViewer('qtcoin')
    
    # set collision checker to Bullet (default collision checker might not recognize cylinder collision for Ubuntu) (causes issues when moving joints)
    collisionChecker = RaveCreateCollisionChecker(env, 'fcl_')
    env.SetCollisionChecker(collisionChecker)
    
    robot = env.GetRobots()[0]

    bounds = np.array([[-2.5, -2.5, -3.14], [2.5, 2.5, 3.14]]) # all random envs are built using the same base world, thus they have the same world bounds
    envmin = np.array([-2.5, -2.5, -2.5])
    envmax = np.array([2.5, 2.5, 2.5])

    generator = InputGenerator(robot, envnum, bounds, [envmin, envmax], runnum)
    generator.generate(goal_loc)
