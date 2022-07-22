import time
import os, sys
import pickle
import openravepy
import numpy as np
from openravepy import *
sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
import fetch

class InputGenerator():

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
        self.fullpath = None
        self.xyzfullpath = None
        self.goal_loc = None
        self.setupRobot()

    def setPixelLocations(self, pixel):
        pixminx = self.bounds[0][0] + (pixel[0] * self.pixeldims[0])
        pixmaxx = self.bounds[0][0] + ((pixel[0] + 1) * self.pixeldims[0])
        
        pixminy = self.bounds[0][1] + (pixel[1] * self.pixeldims[1])
        pixmaxy = self.bounds[0][1] + ((pixel[1] + 1) * self.pixeldims[1])
        
        pixminz = self.bounds[0][2] + (pixel[2] * self.pixeldims[2])
        pixmaxz = self.bounds[0][2] + ((pixel[2] + 1) * self.pixeldims[2])

        return [(pixminx, pixminy, pixminz), (pixmaxx, pixmaxy, pixmaxz)]

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
        # Build fetch location tensor
        # fetch location normalized
        fetch_transform = self.robot.GetTransform()
        fetch_axisAngles = axisAngleFromRotationMatrix(self.robot.GetTransform())
        fetch_xyz = self.robot.GetTransform()[:3,3]
        
        # x,y,z,theta1,theta2,theta3
        fetch_loc_tensor = np.zeros((self.labelsize, self.labelsize, self.labelsize, 6), dtype=np.float32) 
        for i, loc_val in enumerate(fetch_xyz):
            fetch_loc_tensor[:,:,:,i] = (loc_val - self.bounds[0][i]) / (self.bounds[1][i] - self.bounds[0][i])

        for i, rot_val in enumerate(fetch_axisAngles):
            fetch_loc_tensor[:,:,:,i+3] = (rot_val - (-np.pi)) / (2*np.pi)

        # Build environment tensor
        env_tensor = None
        sample_file = os.path.join(INPDIR, str(self.envnum) + '.1.npy')
        if os.path.isfile(sample_file):
            # The env remains the same for different data points so we don't need to regenerate it everytime
            # We can read it from an already generated one
            sample_data = np.load(sample_file)
            env_tensor = sample_data[:,:,:,:3]
        else:
            oldCC = self.env.GetCollisionChecker()
            self.env.Remove(self.robot)
            collisionChecker = RaveCreateCollisionChecker(self.env, 'CacheChecker') # bullet CacheChecker pqp
            self.env.SetCollisionChecker(collisionChecker)
            # binary env image 0->obstacle 1->free space
            env_tensor = np.zeros((self.labelsize,self.labelsize, self.labelsize, 3), dtype=np.float32)
            for i in xrange(self.labelsize):
                for j in xrange(self.labelsize):
                    for k in xrange(self.labelsize):
                        if self.pixelInObstacle((i, j, k)):
                            env_tensor[i][j][k] = (0, 0, 0)
                        else:
                            env_tensor[i][j][k] = (1, 1, 1)
            self.env.SetCollisionChecker(oldCC)
        
        # Build goal configuration tensor
        # goal tensor normalized
        total_dofs = self.robot.GetActiveDOF()
        goal_tensor = np.ones((self.labelsize, self.labelsize, self.labelsize, total_dofs), dtype=np.float32)
        dof_llimits = self.robot.GetActiveDOFLimits()[0]
        dof_ulimits = self.robot.GetActiveDOFLimits()[1]
        for i, dof_value in enumerate(self.goal_loc):
            goal_tensor[:,:,:,i] = (dof_value - dof_llimits[i]) / (dof_ulimits[i] - dof_llimits[i])

        #input tensor
        channels = [env_tensor, goal_tensor, fetch_loc_tensor]
        inp = np.concatenate(channels, axis=3)

        # validate
        assert np.sum(np.isnan(inp)) == 0
        assert len(inp.shape) == 4
        assert inp.shape[0] == self.labelsize and inp.shape[1] == self.labelsize and inp.shape[2] == self.labelsize
        assert inp.shape[3] == 17 # should have 7 channels (0-2: env image, 3-10: goal dof values, 11-16: fetch location)
        for x in range(inp.shape[3]): # all channels should be normalized
            assert inp[:,:,:,x].min() >= 0 and inp[:,:,:,x].max() <= 1.0
        
        # inppath = os.path.join(INPDIR, '{envnum}.{runnum}.npy'.format(envnum=self.envnum, runnum=self.runnum))
        inppath = os.path.join('datatest', '{envnum}.{runnum}.npy'.format(envnum=self.envnum, runnum=self.runnum))

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
        elif self.robot.GetName() == 'fetch':
            llimits = self.robot.GetDOFLimits()[0]
            ulimits = self.robot.GetDOFLimits()[1]
            # DOFs 2, 12, 14 are circular and need their limits to be set to [-3.14, 3.14] as stated by planner
            llimits[2] = llimits[12] = llimits[14] = -np.pi
            ulimits[2] = ulimits[12] = ulimits[14] = np.pi
            self.robot.SetDOFLimits(llimits, ulimits)

    def generate(self, goal_loc):
        '''
        generates label and input tensors
        '''
        self.goal_loc = goal_loc

        print('creating input image')
        inpstarttime = time.time()
        env_tensor, goal_tensor = self.buildInput()
        print('input creation took: ' + str(time.time()-inpstarttime))

if __name__ == '__main__':

    # envPath = sys.argv[1]
    # envnum = float(sys.argv[2])
    # runnum = int(sys.argv[3])
    # visualize = False

    envPath = 'envs/3d/7.3.dae'
    envnum = 7.3
    runnum = 1
    visualize = True

    # goal_loc = [2.,   1.,   1.57, 0.  ] # 8.0.1
    # goal_loc = [-2., -1.,  0.,  0.] # 8.0.2
    # goal_loc = [-2.,  1. , 0.,  0.] # 8.0.3
    # goal_loc = [ 2., -1.,  0.,  0.] # 8.0.4
    # goal_loc = [-0.96970837, -0.82476306, -2.42265842,  0.55285693] # 10.2.1
    # goal_loc = [2.,   1.6,  1.57, 0.  ] # 21.0.1
    # goal_loc = [ 0.19518077, -0.40563759, -0.8       ,  0.8       ,  1.71979317, -0.66500785,  0.94405181, -2.43731356] # 7.1.1 (kitchen)
    # goal_loc = [ 2.60995822e-01, -5.67540971e-02, -5.00000000e-01,  0.00000000e+00, 9.33321341e-02,  1.33919441e-07,  1.97746408e+00, -1.60419046e+00] # 7.2 (factory - machine3)
    goal_loc = [ 0.19275635, -0.17946221,  0.4, 0.5, 0.86871102, -0.35622208, -1.1734052, -0.34213511] # 7.3 (factory - machine4)


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
    bounds = [[-2.5, -2.5, -0.001], [2.5, 2.5, 2.5]]
    # bounds = [[0, 0, -.1], [3., 4., 2.5]]

    DATADIR = os.path.join('data', 'env' + str(envnum), 'data')
    INPDIR = os.path.join('data', 'env' + str(envnum), 'inp')
    if not os.path.isdir(INPDIR):
        os.makedirs(INPDIR)
    LBLDIR = os.path.join('data', 'env' + str(envnum), 'lbl')
    if not os.path.isdir(LBLDIR):
        os.makedirs(LBLDIR)

    print('*********processing {envnum}.{runnum}*********'.format(envnum=str(envnum), runnum=str(runnum)))
    generator = InputGenerator(env, envnum, runnum, bounds, 64)
    generator.generate(goal_loc)
