# adapted from Simple Navigation OpenRAVE example, Rosen Diankov (rosen.diankov@gmail.com)
# http://openrave.org/docs/0.8.2/_modules/openravepy/examples/simplenavigation/#SimpleNavigationPlanning
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

datasetsize = 1
pathsperimage = 100
np.random.seed(None)

class MotionPlansGenerator:
    
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

    def setPixelLocations(self, pixel):
        pixminx = self.bounds[0][0] + (pixel[1] * self.pixelwidth)
        pixminy = self.bounds[1][1] - ((pixel[0] + 1) * self.pixelwidth)
        pixmaxx = self.bounds[0][0] + ((pixel[1] + 1) * self.pixelwidth)
        pixmaxy = self.bounds[1][1] - (pixel[0] * self.pixelwidth)
        
        return [(pixminx, pixminy), (pixmaxx, pixmaxy)]

    def getStartGoal(self, numpaths):
        sd1_value = np.random.uniform(-np.pi,np.pi)
        sd2_value = np.random.uniform(-np.pi/2,np.pi/2)
        gd1_value = np.random.uniform(-np.pi,np.pi)
        gd2_value = np.random.uniform(-np.pi/2,np.pi/2)

        if self.envnum == 5.1:
            s_yvalue = np.random.uniform(0,1)
            g_yvalue = np.random.uniform(0,1)
            if True:
                start = [1.25, s_yvalue, sd1_value, sd2_value]
                goal = [-1.25, g_yvalue, gd1_value, gd2_value]
            else:
                start = [-1.25, s_yvalue, sd1_value, sd2_value]
                goal = [1.25, g_yvalue, gd1_value, gd2_value]

        elif self.envnum == 5.2:
            s_yvalue = np.random.uniform(-0.25,0.75)
            g_yvalue = np.random.uniform(-0.25,0.75)
            if True:
                start = [1.3,s_yvalue, sd1_value, sd2_value]
                goal = [-1.3,g_yvalue, gd1_value, gd2_value]
            else:
                start = [-1.3,s_yvalue, sd1_value, sd2_value]
                goal = [1.3,g_yvalue, gd1_value, gd2_value]

        elif self.envnum == 8.0:
            start = [1.5,0.59,sd1_value, sd2_value]
            goal = [-1.5,0.59,gd1_value, gd2_value]

        elif self.envnum == 9.1:
            if True:
                start = [-7,-7,0]
                goal = [-7,5.5,0]
            else:
                start = [-0.5,6.5,0]
                goal = [-3.5,6.5,0]

        elif self.envnum == 9.2:
            if True:
                start = [0.0205585116,0.593383701,-0.114082917,2.03588441,0,0.518159381,3,-0.8,0.14,0.01]
                goal = [0.0205585116,0.593383701,-0.114082917,2.03588441,0,0.518159381,3,-1.76,-1.73,0.0]
            else:
                start = [0.0205585116,0.593383701,-0.114082917,2.03588441,0,0.518159381,3,-1.76,-1.73,0.0]
                goal = [0.0205585116,0.593383701,-0.114082917,2.03588441,0,0.518159381,3,-0.8,0.14,0.01]

        elif self.envnum == 10.0:
            startx = np.random.uniform(0.3, 1.2)
            starty = np.random.uniform(-1.2, 1.2)
            goalx = np.random.uniform(-1.2, -0.3)
            goaly = np.random.uniform(-1.2, 1.2)
            start = [startx, starty, np.random.uniform(-np.pi, np.pi)]
            goal = [goalx, goaly, np.random.uniform(-np.pi, np.pi)]

        elif self.envnum == 17.0:
            s_yvalue = np.random.uniform(-0.5,0.5)
            g_yvalue = np.random.uniform(-0.5,0.5)
            if True:
                start = [1.3, s_yvalue, sd1_value, sd2_value]
                goal = [-1.3, g_yvalue, gd1_value, gd2_value]
            else:
                start = [-1.3, s_yvalue, sd1_value, sd2_value]
                goal = [1.3, g_yvalue, gd1_value, gd2_value]
        
        elif self.envnum == 17.1 or self.envnum == 18.0 or self.envnum == 18.1:
            s_yvalue = np.random.uniform(-0.5,0.5)
            g_yvalue = np.random.uniform(-0.5,0.5)
            if True:
                start = [1.4, s_yvalue, sd1_value, sd2_value]
                goal = [-1.4, g_yvalue, gd1_value, gd2_value]
            else:
                start = [-1.4, s_yvalue, sd1_value, sd2_value]
                goal = [1.4, g_yvalue, gd1_value, gd2_value]

        elif self.envnum == 17.2:
            s_xvalue = np.random.uniform(-2.0, 0.)
            g_xvalue = np.random.uniform(-2.0, 0.)
            s_yvalue = np.random.uniform(0.8, 1.8)
            g_yvalue = np.random.uniform(-1.8, -0.8)
            if True:
                start = [s_xvalue, s_yvalue, sd1_value, sd2_value]
                goal = [g_xvalue, g_yvalue, gd1_value, gd2_value]
            else:
                start = [g_xvalue, g_yvalue, gd1_value, gd2_value]
                goal = [s_xvalue, s_yvalue, sd1_value, sd2_value]

        elif self.envnum == 18.2 or self.envnum == 22.0:
            s_yvalue = np.random.uniform(-0.5,0.5)
            g_yvalue = np.random.uniform(-0.5,0.5)
            if True:
                start = [1.6, s_yvalue, sd1_value, sd2_value]
                goal = [-1.6, g_yvalue, gd1_value, gd2_value]
            else:
                start = [-1.6, s_yvalue, sd1_value, sd2_value]
                goal = [1.6, g_yvalue, gd1_value, gd2_value]

        elif self.envnum == 19.0:
            s_xvalue = np.random.uniform(0.1, 2.0)
            s_yvalue = np.random.uniform(-2.0, -0.3)
            g_xvalue = np.random.uniform(-2.5, 2.5)
            g_yvalue = np.random.uniform(-2.5, 2.5)
            if True:
                start = [s_xvalue, s_yvalue, sd1_value, sd2_value]
                goal = [g_xvalue, g_yvalue, gd1_value, gd2_value]
            else:
                start = [g_xvalue, g_yvalue, gd1_value, gd2_value]
                goal = [s_xvalue, s_yvalue, sd1_value, sd2_value]

        elif self.envnum == 19.1:
            s_xvalue = np.random.uniform(-2.2, -1.5)
            s_yvalue = np.random.uniform(-2.3, -0.3)
            g_xvalue = np.random.uniform(-2.5, 2.5)
            g_yvalue = np.random.uniform(-2.5, 2.5)
            if True:
                start = [s_xvalue, s_yvalue, sd1_value, sd2_value]
                goal = [g_xvalue, g_yvalue, gd1_value, gd2_value]
            else:
                start = [g_xvalue, g_yvalue, gd1_value, gd2_value]
                goal = [s_xvalue, s_yvalue, sd1_value, sd2_value]
        
        elif self.envnum == 23.0:
            s_xvalue = np.random.uniform(-2.1, -1)
            s_yvalue = np.random.uniform(0.1, 2.0)
            g_xvalue = np.random.uniform(-2.1, -1.5)
            g_yvalue = np.random.uniform(-2.3, -0.6)
            if True:
                start = [s_xvalue, s_yvalue, sd1_value, sd2_value]
                goal = [g_xvalue, g_yvalue, gd1_value, gd2_value]
            else:
                start = [g_xvalue, g_yvalue, gd1_value, gd2_value]
                goal = [s_xvalue, s_yvalue, sd1_value, sd2_value]
        elif self.envnum == 21.0:
            s_xvalue = np.random.uniform(1.4, 2.5)
            s_yvalue = np.random.uniform(0, 2.5)
            g_xvalue = np.random.uniform(self.bounds[0][0],self.bounds[1][0])
            g_yvalue = np.random.uniform(self.bounds[0][1],self.bounds[1][1])
            if True:
                start = [s_xvalue, s_yvalue, sd1_value, sd2_value]
                goal = [g_xvalue, g_yvalue, gd1_value, gd2_value]
            else:
                start = [g_xvalue, g_yvalue, gd1_value, gd2_value]
                goal = [s_xvalue, s_yvalue, sd1_value, sd2_value]
        else:
            while True:
                startx = np.random.uniform(self.bounds[0][0],self.bounds[1][0])
                starty = np.random.uniform(self.bounds[0][1],self.bounds[1][1])
                goalx = np.random.uniform(self.bounds[0][0],self.bounds[1][0])
                goaly = np.random.uniform(self.bounds[0][1],self.bounds[1][1])
                start = [startx, starty, sd1_value, sd2_value]
                goal = [goalx, goaly, gd1_value, gd2_value]

                if (start[0] > 0 and start[1] > 0) and (goal[0] > 0 and goal[1] > 0): # both in Q1
                    continue
                elif (start[0] < 0 and start[1] > 0) and (goal[0] < 0 and goal[1] > 0): # both in Q2
                    continue
                elif (start[0] < 0 and start[1] < 0) and (goal[0] < 0 and goal[1] < 0): # both in Q3
                    continue
                elif (start[0] > 0 and start[1] < 0) and (goal[0] > 0 and goal[1] < 0): # both in Q4
                    continue
                else:
                    break

        return start, goal

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
                
                count += 1 # x, y location
                
                # rotation(theta) bins
                for j in range(n_bin):
                    if point[2] >= d1_bin_start[j] and point[2] <= d1_bin_end[j]:
                        dof_count[0][j] += 1
                
                # alpha bins
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

    def buildLabel(self, fullpath): 
        '''
        build label images manually through prodding robot environment for path pixel-path intersections
        generates images for each channel
        '''
        lbl = np.zeros((224,224,3), dtype=float)

        xyfullpath = []
        for path in fullpath:
            xypath = [point[:2] for point in path]
            xyfullpath.append(xypath)

        for row in xrange(lbl.shape[0]):
            for col in xrange(lbl.shape[1]):
                lbl[row][col] = self.pixelInPath((row, col), fullpath, xyfullpath, lbl.shape[2])

        for channel in range(lbl.shape[2]):
            imgpath = os.path.join('data', 'env' + str(self.envnum), 'lbl', str(channel) + '.png')
            imsave(imgpath, lbl[:,:,channel], 'PNG')

            # ax = sns.heatmap(lbl[:,:,channel], linewidth=0.5)
            # plt.imshow(lbl[:,:,channel], cmap='viridis')
            # plt.colorbar()
            # plt.show()

        npypath = os.path.join('data', 'env' + str(self.envnum), 'data', str(self.envnum) + '.' + str(self.runnum) + '_lbl.npy')
        np.save(npypath, lbl)
        return
 
    def pixelInObstacle(self, pindex, robot=False):
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
        
        # move robot to pixel center and see if a collision arrises
        collision = False
        if robot:
            with self.env:
                self.robot.SetActiveDOFValues([aveX, aveY, 0])

            if self.env.CheckCollision(self.robot):
                collision = True

            with self.env:
                if self.envnum == 9.2 or self.envnum == 10.2:
                    self.robot.SetActiveDOFValues([0,0,0,0,0,0,0]+[15, 15, 0])
                else:
                    self.robot.SetActiveDOFValues([15, 15, 0])

        else:
            body = openravepy.RaveCreateKinBody(self.env, '')
            body.SetName('tracer') # make height of env or robot
            body.InitFromBoxes(np.array([[aveX, aveY, 1.05, self.pixelwidth, self.pixelwidth, 1.0]]), True) # make sure z-coordinate is bounded by shortest wall height
            self.env.AddKinBody(body)
            self.env.UpdatePublishedBodies()
            #pdb.set_trace()
            if self.env.CheckCollision(body): 
                self.env.Remove(body)
                collision = True
            else:
                self.env.Remove(body)
    
        return collision		    	

    def buildEnv(self, imgpath, robot):
        '''
        build environment images manually through prodding robot environment
        '''
        oldCC = self.env.GetCollisionChecker()
        
        if not robot:
            self.env.Remove(self.robot)
            collisionChecker = RaveCreateCollisionChecker(self.env, 'CacheChecker') # bullet CacheChecker pqp
            self.env.SetCollisionChecker(collisionChecker)
        else:
            prev = self.robot.GetActiveDOFValues()
            with self.env:
                if self.envnum == 9.2 or self.envnum == 10.2:
                    self.robot.SetActiveDOFValues([0,0,0,0,0,0,0]+[15, 15, 0])
                else:
                    self.robot.SetActiveDOFValues([15, 15, 0])
        
        img = np.zeros((224,224,3), dtype=int)
        #pdb.set_trace()
        for row in xrange(224):
            for col in xrange(224):
                if self.pixelInObstacle((row, col), robot=robot):
                    img[row][col] = (0,0,0)
                else:
                    img[row][col] = (255, 255, 255)
        
        imsave(imgpath, img, 'PNG')
        
        if robot:
            with self.env:
                self.robot.SetActiveDOFValues(prev)
        #pdb.set_trace()
        self.env.SetCollisionChecker(oldCC)

    def buildSM(self, imgpath, gazepath, lblpath, smpath):
        '''
        build saliency map from gaze labels
        '''
        oi = OpencvIo()
        gaze = oi.imread(lblpath)
        sm = SaliencyMap(gaze)
        oi.imwrite(sm.map, gazepath)
        oi.imwrite(sm.map, smpath)

        # threshold SM
        with open(gazepath, 'r+b') as f:
            with PIL.Image.open(f).convert('RGB') as image:
                with PIL.Image.open(imgpath).convert('RGB') as envimage:
                    for width in xrange(image.size[0]):
                        for length in xrange(image.size[1]):
                            pixel = image.getpixel((width, length))
                            envpixel = envimage.getpixel((width, length)) 

                            # putpixel() is slow, learn to use paste for single pixel placement if possible
                            if envpixel == (0,0,0):
                                image.putpixel((width, length), (0,0,0))
                            elif pixel[0] <= 100: 
                                image.putpixel((width, length), (0, 0, 0))
                            else:
                                image.putpixel((width, length), (255, 255, 255))

                    image.save(gazepath, image.format)

    def buildMatrix(self):
        gazedir = os.path.join('data', 'env' + str(self.envnum), 'gaze')
        channels = []
        for image in xrange(len(glob.glob(gazedir+'/*.png'))):
            lblpath = os.path.join(gazedir, str(image) + '.png')
            im = cv2.imread(lblpath)
            channels.append(im[:,:,0])
        image = np.stack(channels, axis=2)
        assert image.shape[2] == 3
        matrixpath = os.path.join(gazedir, str(self.envnum) + '.' + str(self.runnum) + '.npy')
        np.save(matrixpath, image)

    def setupRobot(self):
        if self.OMPL:
            if self.robot.GetName() == 'robot1' or self.robot.GetName() == "bot":
                self.robot.SetDOFVelocities([0.5,0.5,0.5])
                self.robot.SetActiveDOFs([0,1,2])
                self.robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
            elif self.robot.GetName() == 'lshaped':
                self.robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
                self.robot.SetActiveDOFs([0,1,2,4])
                self.robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])
        else:
            self.robot.SetAffineTranslationLimits(self.limits[0], self.limits[1])
            self.robot.SetAffineTranslationMaxVels([0.5,0.5,0.5])
            self.robot.SetAffineRotationAxisMaxVels(np.ones(4))
            if self.envnum == 9.2 or self.envnum == 10.2:
                self.robot.SetActiveDOFs([0,1,2,3,4,5,6],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1])
                self.robot.Grab(self.env.GetKinBody('mug6'))
            else:
                self.robot.SetActiveDOFs([],DOFAffine.X|DOFAffine.Y|DOFAffine.RotationAxis,[0,0,1]) # joint DOFs, affine DOFs, axis of rotation

    def generate(self):
        mpstarttime = time.time()
        size = 0
        numpaths = 1
        trace = []
        path = []
        fullpath = []
        
        self.setupRobot()
        
        #pdb.set_trace()
        # motion and tracing
        trajectory_info = []

        # Fix goal state
        while True: 
            _, goal = self.getStartGoal(self.runnum)
            with self.env:
                with self.robot:
                    self.robot.SetActiveDOFValues(goal)
                    if not self.env.CheckCollision(self.robot):
                        break

        while size < datasetsize:
            print("Path: ", numpaths)
            print("Initializing start/goal states..")
            while True: # Get start state
                start, _ = self.getStartGoal(self.runnum)
                with self.env:
                    with self.robot:
                        self.robot.SetActiveDOFValues(start)
                        if not self.env.CheckCollision(self.robot):
                            break

            self.robot.SetActiveDOFValues(start)
            print('planning from: ' + str(start) + '\nto: ' + str(goal))
            planningstarttime = time.time()
            if self.OMPL: # Plan path
                with self.robot:
                    planner = RaveCreatePlanner(self.env, 'OMPL_RRTConnect')
                    params = Planner.PlannerParameters()
                    params.SetRobotActiveJoints(self.robot)
                    params.SetConfigAccelerationLimit(self.robot.GetActiveDOFMaxAccel())
                    params.SetConfigVelocityLimit(self.robot.GetActiveDOFMaxVel())
                    params.SetInitialConfig(start)
                    params.SetGoalConfig(goal)
                    params.SetExtraParameters('<range>0.01</range>')
                    params.SetExtraParameters('<time_limit>180</time_limit>')
                    planner.InitPlan(self.robot, params)
                    traj = RaveCreateTrajectory(self.env, '')
                    with CollisionOptionsStateSaver(self.env.GetCollisionChecker(), CollisionOptions.ActiveDOFs):
                        result = planner.PlanPath(traj)

                if not result == PlannerStatus.HasSolution:
                    print 'retrying...'
                    continue
                else:
                    result = planningutils.RetimeTrajectory(traj)
                    if not result == PlannerStatus.HasSolution:
                        print 'retrying...'
                        continue
            else:
                try:
                    traj = self.basemanip.MoveActiveJoints(goal=goal, maxiter=200000, steplength=0.1, execute=False, outputtrajobj=True) # path trajectory information
                except:
                    print 'retrying...'
                    continue

                if traj is None: # allows trajectory if no joint collisions
                    print 'retrying...'
                    continue
            
            print("motion planning took: ", time.time() - planningstarttime)
            # Execute trajectory
            if self.env.GetViewer() is not None:
                with self.robot:
                    self.robot.GetController().SetPath(traj)
                self.robot.WaitForController(0)

            # Save points traversed in the trajectory
            totaldof = len(self.robot.GetActiveDOFValues())
            movepath = []
            if self.OMPL:
                sleeptime = 0.01
            else:
                sleeptime = 0.01
            starttime = time.time()
            while time.time()-starttime <= traj.GetDuration():
                curtime = time.time() - starttime
                trajdata = traj.Sample(curtime) 
                movepath.append(trajdata[0:totaldof])
                # path.append(list(trajdata[totaldof-3:totaldof-1]))
                time.sleep(sleeptime)

            cleaned_movepath = []
            for p in movepath:
                self.robot.SetActiveDOFValues(p[:totaldof])
                if not self.env.CheckCollision(self.robot):
                    cleaned_movepath.append(p)
            
            # Add end location (Sampled trajectory might not have final end point)
            trajdata = traj.Sample(traj.GetDuration())
            cleaned_movepath.append(trajdata[0:totaldof])
            # path.append(list(trajdata[0:totaldof-1]))
            print("Sampled path length: ", len(cleaned_movepath))
            print("Running env: ", self.envnum, self.runnum)
            fullpath.append(cleaned_movepath)
            trajectory_info.append({
                'start': start,
                'goal': goal,
                'traj': traj.serialize(),
                'path': cleaned_movepath
            })

            # trace and move robot
            if self.env.GetViewer() is not None:
                for point in cleaned_movepath:
                    # with self.env: # lock environment when accessing robot
                        self.robot.SetActiveDOFValues(point)
                        # position (x, y)
                        # trace.append(self.env.plot3(points=[point[totaldof-3],point[totaldof-2],0.25], pointsize=0.03, colors=np.array((0,1,0,0.5)), drawstyle=1)) # tranlucent green
                        if point[2] >= -np.pi/4 and point[2] <= np.pi/4:
                            trace.append(self.env.plot3(points=[point[0],point[1],0.25], pointsize=0.03, colors=np.array((1,0,0,0.5)), drawstyle=1)) # tranlucent red
                        elif point[2] <= -np.pi/4 and point[2] >= -3*np.pi/4:
                            trace.append(self.env.plot3(points=[point[0],point[1],0.25], pointsize=0.03, colors=np.array((0,0,1,0.5)), drawstyle=1)) # tranlucent blue
                        elif point[2] >= np.pi/4 and point[2] <= 3*np.pi/4:
                            trace.append(self.env.plot3(points=[point[0],point[1],0.25], pointsize=0.03, colors=np.array((0,1,0,0.5)), drawstyle=1)) # tranlucent green
                        else:
                            trace.append(self.env.plot3(points=[point[0],point[1],0.25], pointsize=0.03, colors=np.array((1,1,0,0.5)), drawstyle=1)) # tranlucent yellow
            
            # save multi-path labels 
            if numpaths < pathsperimage:
                numpaths += 1
            else:
                # reset for next run
                fullpath = []
                trace = []
                size += 1
                numpaths = 1
                mpstarttime = time.time()

            # wait 
            print '\nwaiting for controller'
            self.robot.WaitForController(0)
            print 'controller updated'

            # reset for next path
            movepath = []		 
            path = []

        traj_info_path = os.path.join('data', 'env' + str(self.envnum), 'data', 
                                        str(self.envnum) + '.' + str(self.runnum) + '_traj.pkl')
        pickle.dump(trajectory_info, open(traj_info_path, 'wb'))
        
        return