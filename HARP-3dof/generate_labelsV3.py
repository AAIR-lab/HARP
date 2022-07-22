import time
import os, sys
import pickle
import openravepy
import numpy as np
from openravepy import *
sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
# import fetch

'''
generates inputs and (number_of_dofs x number_of_bins) shaped labels
'''

class LabelGenerator():

    def __init__(self, env, envnum, runnum, bounds, labelsize=224):      
        self.env = env
        self.envnum = envnum
        self.runnum = runnum
        self.bounds = bounds
        self.robot = env.GetRobots()[0]
        self.bounds = bounds
        self.labelsize = labelsize
        self.pixeldims = [0,0,0]
        self.pixeldims[0] = (bounds[1][0] - bounds[0][0]) / self.labelsize # x
        self.pixeldims[1] = (bounds[1][1] - bounds[0][1]) / self.labelsize # y
        self.pixelBounds = np.fromfunction(lambda i, j: self.setPixelLocations((i, j)), (self.labelsize,self.labelsize), dtype=int)
        self.number_of_bins = 20
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
        
        return [(pixminx, pixminy), (pixmaxx, pixmaxy)]

    def pixelInObstacle(self, pindex):
        '''
        see if moving pixel-sized cube to centroid of pixel bound causes collision; return True if so, else False
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
        body.InitFromBoxes(np.array([[aveX, aveY, 1.05, self.pixeldims[0], self.pixeldims[1], 1.0]]), True) # make sure z-coordinate is bounded by shortest wall height
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
        
        # x,y,z,theta1,theta2,theta3

        # Build environment tensor

        env_tensor = None
        sample_file = os.path.join(INPDIR, str(self.envnum) + '.1.npy')
        if os.path.isfile(sample_file):
            # The env remains the same for different data points so we don't need to regenerate it everytime
            # We can read it from an already generated one
            sample_data = np.load(sample_file)
            env_tensor = sample_data[:,:,:3]
        else:
            oldCC = self.env.GetCollisionChecker()
            self.env.Remove(self.robot)
            collisionChecker = RaveCreateCollisionChecker(self.env, 'CacheChecker') # bullet CacheChecker pqp
            self.env.SetCollisionChecker(collisionChecker)
            # binary env image 0->obstacle 1->free space
            env_tensor = np.zeros((224,224,3), dtype=np.float32)
            for row in xrange(224):
                for col in xrange(224):
                    if self.pixelInObstacle((row, col)):
                        env_tensor[row][col] = (0, 0, 0) # [0,1]
                        # env_tensor[row][col] = (-1,-1,-1) # [-1,1] 
                    else:
                        env_tensor[row][col] = (1, 1, 1)
            self.env.SetCollisionChecker(oldCC)
        
        # Build goal configuration tensor
        # goal tensor normalized
        total_dofs = self.robot.GetActiveDOF()
        goal_tensor = np.ones((224,224, total_dofs), dtype=np.float32)
        dof_llimits = self.robot.GetActiveDOFLimits()[0]
        dof_ulimits = self.robot.GetActiveDOFLimits()[1]
        for i, dof_value in enumerate(self.goal_loc):
            goal_tensor[:,:,i] = (dof_value - dof_llimits[i]) / (dof_ulimits[i] - dof_llimits[i])

        #input tensor
        channels = [env_tensor, goal_tensor]
        inp = np.concatenate(channels, axis=2)

        # validate
        # assert np.sum(np.isnan(inp)) == 0
        # assert len(inp.shape) == 3


        assert inp.shape[0] == 224 and inp.shape[1] == 224
        assert inp.shape[2] == 7# should have 7 channels (0-2: env image, 3-10: goal dof values, 11-16: fetch location) dimension_of_env + number_of_dofs for goal(incl x and y)
        for x in range(inp.shape[2]): # all channels should be normalized
            assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0
        
        inppath = os.path.join(INPDIR, '{envnum}.{runnum}.npy'.format(envnum=self.envnum, runnum=self.runnum))
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

    def load_motionplans(self):
        print('Loading motion plans..')
        trajectory_info = pickle.load(open(DATADIR + '/{envnum}.{runnum}_traj.pkl'.format(envnum=self.envnum, runnum=self.runnum), 'rb'))
        self.goal_loc = trajectory_info[0]['goal']
        fullpath = []
        for trajectory in trajectory_info:
            fullpath.append(trajectory['path'])
        self.fullpath = fullpath
    
    def get_bin(self,dof_bins,value,dof_number):
        for j in range(self.number_of_bins):
            if value >= dof_bins[dof_number]["bin_start"][j] and value < dof_bins[dof_number]["bin_end"][j]:
                return j
        return -1

    def buildLabel(self):  

        # create dof bins
        n_bin = self.number_of_bins
        llimits = self.robot.GetActiveDOFLimits()[0]
        ulimits = self.robot.GetActiveDOFLimits()[1]
        dof_ranges = ulimits - llimits
        dof_bins = {}
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
        for i in range(len(self.goal_loc)): # iterate over each dof
            for j in range(self.number_of_bins): # count which bin does dof value fall into
                if self.goal_loc[i] >= dof_bins[i]['bin_start'][j] and \
                    self.goal_loc[i] < dof_bins[i]['bin_end'][j]:
                        goal_bins.append(j)
        # print(goal_bins)

        distribution = np.zeros((self.number_of_bins ,self.number_of_bins))

        dof_3_count = np.zeros((distribution.shape[0], distribution.shape[0], self.number_of_bins))
        dof_4_count = np.zeros(dof_3_count.shape)

        for path in self.fullpath:
            flag = np.zeros(distribution.shape)
            for point in path:
                # for i in range(len(point)): # iterate over each dof
                #     for j in range(n_bin): # count which bin does dof value fall into
                #         if point[i] >= dof_bins[i]['bin_start'][j] and \
                #             point[i] < dof_bins[i]['bin_end'][j]:
                #                 if flag[i,j] == 0:  #remove the flag variable to revert the change
                #                     distribution[i, j] += 1
                #                     flag[i,j] == 1
                #                 else:
                #                     distribution[i,j] += 0.05
                x_bin = self.get_bin(dof_bins,point[0],0)
                y_bin = self.get_bin(dof_bins,point[1],1)
                dof_3_bin = self.get_bin(dof_bins,point[2],2)
                dof_4_bin = self.get_bin(dof_bins,point[3],3)
                if flag[x_bin,y_bin] == 0:
                    distribution[x_bin,y_bin] += 1
                    flag[x_bin,y_bin] = 1
                else:
                    # distribution[x_bin,y_bin] += 0.01
                    pass
                dof_3_count[x_bin,y_bin,dof_3_bin] += 1
                dof_4_count[x_bin,y_bin,dof_4_bin] += 1

        dof_3_channel = np.argmax(dof_3_count,axis=2) / float(self.number_of_bins)
        dof_4_channel = np.argmax(dof_4_count,axis=2) / float(self.number_of_bins)

        # print(distribution)
        pd_xy = distribution / len(self.fullpath)
        # print(pd)

        pd = np.array([pd_xy,dof_3_channel,dof_4_channel])
        pd = np.nan_to_num(pd)

        assert pd.min() >= 0.0 and pd.max() <= 1.0
        assert len(pd.shape) == 3
        assert pd.shape[0] == self.robot.GetActiveDOF() - 1 and pd.shape[1] == self.number_of_bins and pd.shape[2] == self.number_of_bins

        lblpath = os.path.join(LBLDIR, str(self.envnum) + '.' + str(self.runnum) + '.npy')
        np.save(lblpath, pd)

    def generate(self):
        '''
        generates label and input tensors
        '''

        print('creating label')
        labelstarttime = time.time()
        lbl = self.buildLabel()
        print('label creation took: ' + str(time.time() - labelstarttime))

        print('creating input image')
        inpstarttime = time.time()
        env_tensor, goal_tensor = self.buildInput()
        print('input creation took: ' + str(time.time()-inpstarttime))

if __name__ == '__main__':

    # print sys.argv
    envPath = sys.argv[1]
    envnum = float(sys.argv[2])
    runnum = int(sys.argv[3])
    visualize = False

    # envPath = 'envs/3d/2.0.dae'
    # envPath = './envs3d/5.1.xml'
    # envnum = 5.1
    # runnum = 1
    # visualize = True

    env = Environment()
    env.Load(envPath)
    if visualize: 
        env.SetViewer('qtcoin')
    
    # set collision checker to Bullet (default collision checker might not recognize cylinder collision for Ubuntu) (causes issues when moving joints)
    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    collisionChecker.SetCollisionOptions(CollisionOptions.Contacts)
    env.SetCollisionChecker(collisionChecker)
    
    # initial_transform = pickle.load(open(os.path.join('envs','3d','fetch_transforms', str(envnum) + '.pkl'), 'rb'))
    # fetch_robot = fetch.FetchRobot(env, initial_transform)
    bounds = [[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]]

    DATADIR = os.path.join('data', 'env' + str(envnum), 'data')
    INPDIR = os.path.join('data', 'env' + str(envnum), 'inp')
    if not os.path.isdir(INPDIR):
        os.makedirs(INPDIR)
    LBLDIR = os.path.join('data', 'env' + str(envnum), 'lbl')
    if not os.path.isdir(LBLDIR):
        os.makedirs(LBLDIR)

    print('*********processing {envnum}.{runnum}*********'.format(envnum=str(envnum), runnum=str(runnum)))
    generator = LabelGenerator(env, envnum, runnum, bounds, 224)
    generator.generate()
