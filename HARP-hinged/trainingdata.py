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

datasetsize = 1
pathsperimage = 50
np.random.seed(None)

class mNavigationPlanningLabels:
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
        if self.envnum == 5.1:
            s_yvalue = np.random.uniform(0,1)
            g_yvalue = np.random.uniform(0,1)
            if numpaths%2 == 0:
                start = [1.25,s_yvalue,0]
                goal = [-1.25,g_yvalue,0]
            else:
                start = [-1.25,s_yvalue,0]
                goal = [1.25,g_yvalue,0]

        elif self.envnum == 5.2:
            s_yvalue = np.random.uniform(-0.25,0.75)
            g_yvalue = np.random.uniform(-0.25,0.75)
            if numpaths%2 == 0:
                start = [1,s_yvalue,0]
                goal = [-1,g_yvalue,0]
            else:
                start = [-1,s_yvalue,0]
                goal = [1,g_yvalue,0]

        # elif self.envnum == 8.0:
        #     start = [1.5,0.59,0]
        #     goal = [-1.5,0.59,0]

        elif self.envnum == 9.1:
            if numpaths%2 == 0:
                start = [-7,-7,0]
                goal = [-7,5.5,0]
            else:
                start = [-0.5,6.5,0]
                goal = [-3.5,6.5,0]

        elif self.envnum == 9.2:
            if numpaths%2 == 0:
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

        else:
            while True:
                startx = np.random.uniform(self.bounds[0][0],self.bounds[1][0])
                starty = np.random.uniform(self.bounds[0][1],self.bounds[1][1])
                goalx = np.random.uniform(self.bounds[0][0],self.bounds[1][0])
                goaly = np.random.uniform(self.bounds[0][1],self.bounds[1][1])
                start = [startx, starty, np.random.uniform(-np.pi, np.pi)]
                goal = [goalx, goaly, np.random.uniform(-np.pi, np.pi)]

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

    def pixelInPath(self, pindex, fullpath, xyfullpath):
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

        # count = 0
        count = np.array([0, 0, 0, 0, 0])
        for i, path in enumerate(fullpath):
            # xypath = [point[:2] for point in path]
            xypath = xyfullpath[i]
            closestindex = distance.cdist([(aveX,aveY)], xypath).argmin()
            point = path[closestindex]
            if self.pointInBounds(point, b):
                
                count[0] += 1 # x, y location
                
                # rotation(theta) bins
                if point[2] >= -np.pi/4 and point[2] <= np.pi/4:
                    count[1] += 1
                elif point[2] <= -np.pi/4 and point[2] >= -3*np.pi/4:
                    count[2] += 1
                elif point[2] >= np.pi/4 and point[2] <= 3*np.pi/4:
                    count[3] += 1
                elif (point[2] <= -3*np.pi/4 or point[2] >= 3*np.pi/4):
                    count[4] += 1
                
        pvalue = (255*count/len(fullpath))
        return tuple(pvalue) #(pvalue, pvalue, pvalue)

    def buildLabel(self, fullpath): 
        '''
        build label images manually through prodding robot environment for path pixel-path intersections
        generates images for each channel
        '''
        lbl = np.zeros((224,224,5), dtype=int)

        xyfullpath = []
        for path in fullpath:
            xypath = [point[:2] for point in path]
            xyfullpath.append(xypath)

        for row in xrange(lbl.shape[0]):
            for col in xrange(lbl.shape[1]):
                lbl[row][col] = self.pixelInPath((row, col), fullpath, xyfullpath)

        for channel in range(lbl.shape[2]):
            imgpath = os.path.join('data', 'env' + str(self.envnum), 'lbl', str(channel) + '.png')
            imsave(imgpath, lbl[:,:,channel], 'PNG')

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
        assert image.shape[2] == 5
        matrixpath = os.path.join(gazedir, str(self.envnum) + '.' + str(self.runnum) + '.npy')
        np.save(matrixpath, image)

    def setupRobot(self):
        if self.OMPL:
            if self.robot.GetName() == 'robot1':
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

    def performNavigationPlanning(self):
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
        while size < datasetsize:
            print("\nPath: ", numpaths)
            print("Initializing start/goal states..")
            while True: # Get initial and start states
                start, goal = self.getStartGoal(numpaths)
                with self.env:
                    with self.robot:
                        self.robot.SetActiveDOFValues(start)
                        if not self.env.CheckCollision(self.robot):
                            self.robot.SetActiveDOFValues(goal)
                            if not self.env.CheckCollision(self.robot):
                                break

            self.robot.SetActiveDOFValues(start)
            print('planning from: ' + str(start) + '\nto: ' + str(goal))
            
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
            
            # Execute trajectory
            with self.robot:
                self.robot.GetController().SetPath(traj)
            self.robot.WaitForController(0)

            # Save points traversed in the trajectory
            totaldof = len(self.robot.GetActiveDOFValues())
            movepath = []
            if self.OMPL:
                sleeptime = 0.005
            else:
                sleeptime = 0.01
            starttime = time.time()
            while time.time()-starttime <= traj.GetDuration():
                curtime = time.time() - starttime
                trajdata = traj.Sample(curtime) 
                movepath.append(trajdata[0:totaldof])
                path.append(list(trajdata[totaldof-3:totaldof-1]))
                time.sleep(sleeptime)
            
            # Add end location (Sampled trajectory might not have final end point)
            trajdata = traj.Sample(traj.GetDuration())
            movepath.append(trajdata[0:totaldof])
            path.append(list(trajdata[totaldof-3:totaldof-1]))
            print("Sampled path length: ", len(movepath))
            fullpath.append(movepath)
            trajectory_info.append({
                'start': start,
                'goal': goal,
                'traj': traj.serialize(),
                'path': movepath
            })

            # trace and move robot
            for point in movepath:
                with self.env: # lock environment when accessing robot
                    self.robot.SetActiveDOFValues(point)
                    # position (x, y)
                    # trace.append(self.env.plot3(points=[point[totaldof-3],point[totaldof-2],0.25], pointsize=0.03, colors=np.array((0,1,0,0.5)), drawstyle=1)) # tranlucent green
                    if point[2] >= -np.pi/4 and point[2] <= np.pi/4:
                        trace.append(self.env.plot3(points=[point[totaldof-3],point[totaldof-2],0.25], pointsize=0.03, colors=np.array((1,0,0,0.5)), drawstyle=1)) # tranlucent red
                    elif point[2] <= -np.pi/4 and point[2] >= -3*np.pi/4:
                        trace.append(self.env.plot3(points=[point[totaldof-3],point[totaldof-2],0.25], pointsize=0.03, colors=np.array((0,0,1,0.5)), drawstyle=1)) # tranlucent blue
                    elif point[2] >= np.pi/4 and point[2] <= 3*np.pi/4:
                        trace.append(self.env.plot3(points=[point[totaldof-3],point[totaldof-2],0.25], pointsize=0.03, colors=np.array((0,1,0,0.5)), drawstyle=1)) # tranlucent green
                    # elif point[2] <= -np.pi/2 and point[2] >= -np.pi:
                    else:
                        trace.append(self.env.plot3(points=[point[totaldof-3],point[totaldof-2],0.25], pointsize=0.03, colors=np.array((1,1,0,0.5)), drawstyle=1)) # tranlucent yellow
                    # else:
                    #     print(point[2])
            
            # save multi-path labels 
            if numpaths < pathsperimage:
                numpaths += 1
            else:
                # create label without screenshot
                #pdb.set_trace()
                print '\nMP timing: '+str(time.time()-mpstarttime)
                print '\ncreating label: ' + str(self.envnum) + '.' + str(self.runnum)
                # lblpath = os.path.join('data', 'env' + str(self.envnum), 'lbl', str(self.envnum) + str(size) + '.png')
                labelstarttime = time.time()
                self.buildLabel(fullpath)
                print 'label timing: '+str(time.time()-labelstarttime)
                print 'label created'
 
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

        # create env scans
        print '\ncreating environment scans'
        lbldir = os.path.join('data', 'env' + str(self.envnum), 'lbl')
        env = os.path.join('env' + str(self.envnum) + '.' + str(self.runnum) + '.png')
        envstarttime = time.time()
        self.buildEnv(env, robot=False)
        print 'env timing: '+str(time.time()-envstarttime)

        print '\n creating saliency maps and gaze'
        gazestarttime = time.time()
        for image in xrange(len(os.listdir(lbldir))):
            imgpath = os.path.join('data', 'env' + str(self.envnum), 'img', 
                                    str(self.envnum) + '.' + str(self.runnum) + '.png')
            shutil.copy(env, imgpath)
            smpath = os.path.join('data', 'env' + str(self.envnum), 'sm', str(image) + '.png')
            gazepath = os.path.join('data', 'env' + str(self.envnum), 'gaze', str(image) + '.png')
            lblpath = os.path.join('data', 'env' + str(self.envnum), 'lbl', str(image) + '.png')
            self.buildSM(imgpath, gazepath, lblpath, smpath)
        print 'gaze timing: '+str(time.time()-gazestarttime)

        self.buildMatrix()
        os.remove(env)
        print 'scans created'
