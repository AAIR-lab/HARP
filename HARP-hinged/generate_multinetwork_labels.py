import time
import os, sys
import pickle
import openravepy
import numpy as np
from openravepy import *
import cv2
sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
# saliencypath = os.path.join('saliency-map-master', 'src')
sys.path.append('saliency-map-master/src')
from saliency_map import SaliencyMap
# import fetch

'''
generates inputs and (number_of_dofs x number_of_bins) shaped labels
'''

class LabelGenerator():

    def __init__(self, env, envnum, runnum, bounds, labelsize=224, threshold = 0.7 ):      
        self.env = env
        self.envnum = envnum
        self.runnum = runnum
        self.bounds = bounds
        self.robot = env.GetRobots()[0]
        self.bounds = bounds
        self.labelsize = labelsize
        self.threshold = threshold
        self.pixeldims = [0,0,0]
        self.pixeldims[0] = (bounds[1][0] - bounds[0][0]) / self.labelsize # x
        self.pixeldims[1] = (bounds[1][1] - bounds[0][1]) / self.labelsize # y
        self.pixelBounds = np.fromfunction(lambda i, j: self.setPixelLocations((i, j)), (self.labelsize,self.labelsize), dtype=int)
        self.number_of_dof_bins = 10
        self.number_of_xy_bins = self.labelsize
        self.fullpath = None
        self.xyzfullpath = None
        self.goal_loc = None
        self.setupRobot()
        self.number_of_dofs = self.robot.GetActiveDOF()
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
        assert inp.shape[2] == 3 + self.number_of_dofs # should have 7 channels (0-2: env image, 3-10: goal dof values, 11-16: fetch location) dimension_of_env + number_of_dofs for goal(incl x and y)
        for x in range(inp.shape[2]): # all channels should be normalized
            assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0
        
        inppath = os.path.join(INPDIR, '{envnum}.{runnum}.npy'.format(envnum=self.envnum, runnum=self.runnum))
        np.save(inppath, inp)
        return env_tensor, goal_tensor

    def setupRobot(self):
        if self.robot.GetName() == 'robot1' or self.robot.GetName() == "bot":
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
        for j in range(len(dof_bins[dof_number]["bin_start"])):
            if value >= dof_bins[dof_number]["bin_start"][j] and value < dof_bins[dof_number]["bin_end"][j]:
                return j
        return -1

    def buildLabel(self):  

        # create dof bins
        n_bin = self.number_of_dof_bins
        llimits = self.robot.GetActiveDOFLimits()[0]
        ulimits = self.robot.GetActiveDOFLimits()[1]
        dof_ranges = ulimits - llimits
        dof_bins = {}
        for i in range(2):
            dof_bins[i] = {}
            dof_bins[i]['bin_start'] = []
            dof_bins[i]['bin_end'] = []
            dof_bin_range = dof_ranges[i]/self.number_of_xy_bins
            s = llimits[i]
            for j in range(self.number_of_xy_bins):
                dof_bins[i]['bin_start'].append(s)
                dof_bins[i]['bin_end'].append(s + dof_bin_range)
                s += dof_bin_range

        for i in range(2,len(dof_ranges)):
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
            if i < 2: # iterate over each dof
                for j in range(self.number_of_xy_bins): # count which bin does dof value fall into
                    if self.goal_loc[i] >= dof_bins[i]['bin_start'][j] and \
                        self.goal_loc[i] < dof_bins[i]['bin_end'][j]:
                            goal_bins.append(j)
            else:
                for j in range(self.number_of_dof_bins): # count which bin does dof value fall into
                    if self.goal_loc[i] >= dof_bins[i]['bin_start'][j] and \
                        self.goal_loc[i] < dof_bins[i]['bin_end'][j]:
                            goal_bins.append(j)
        # print(goal_bins)

        xy_distribution = np.zeros((self.number_of_xy_bins ,self.number_of_xy_bins))

        dof_3_count = np.zeros((xy_distribution.shape[0], xy_distribution.shape[1], self.number_of_dof_bins))
        dof_4_count = np.zeros(dof_3_count.shape)

        for path in self.fullpath:
            flag = np.zeros(xy_distribution.shape)
            flag_dof_3 = np.zeros(dof_3_count.shape)
            flag_dof_4 = np.zeros(dof_4_count.shape)
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
                    xy_distribution[x_bin,y_bin] += 1.0
                    flag[x_bin,y_bin] = 1
                else:
                    # distribution[x_bin,y_bin] += 0.01
                    pass
                if flag_dof_3[x_bin,y_bin,dof_3_bin] == 0:
                    dof_3_count[x_bin,y_bin,dof_3_bin] += 1.0
                    flag_dof_3[x_bin,y_bin,dof_3_bin] = 1
                if flag_dof_4[x_bin,y_bin,dof_4_bin] == 0:
                    dof_4_count[x_bin,y_bin,dof_4_bin] += 1.0
                    flag_dof_4[x_bin,y_bin,dof_4_bin] = 1

                # dof_4_count[x_bin,y_bin,dof_4_bin] += 1.0

        # dof_3_channel = np.argmax(dof_3_count,axis=2) / float(self.number_of_bins)
        # dof_4_channel = np.argmax(dof_4_count,axis=2) / float(self.number_of_bins)

        # dof_3_count = dof_3_count / float(len(self.fullpath))so I am confident that putting your heads together will resolve this speedily. As a starting 
        # dof_4_count = dof_4_count / float(len(self.fullpath))

        dof_3_channel = np.zeros(dof_3_count.shape)

        for i in range(dof_3_channel.shape[0]):
            for j in range(dof_3_channel.shape[1]):
                max_index = np.argmax(dof_3_count[i,j,:])
                dof_3_channel[i,j,max_index] = 1.0

        dof_4_channel = np.zeros(dof_4_count.shape)

        for i in range(dof_4_channel.shape[0]):
            for j in range(dof_4_channel.shape[1]):
                max_index = np.argmax(dof_4_count[i,j,:])
                dof_4_channel[i,j,max_index] = 1.0

        # dof_3_channel = dof_3_count / np.max(dof_3_count,axis = -1)
        # dof_3_channel = dof_3_channel.astype(np.float32)
        # dof_3_channel = (dof_3_count >= self.threshold).astype(np.float32)
        # dof_4_channel = (dof_4_channel >= self.threshold).astype(np.float32)


        # print(distribution)
        pd_xy = xy_distribution / float(len(self.fullpath))
        # pd_xy = (pd_xy >= self.threshold).astype(np.float32)
        # print(pd)

        pd_xy =  pd_xy.reshape((pd_xy.shape[0],pd_xy.shape[1],1))

        print np.mean(pd_xy)

        
        pd_xy = pd_xy / (np.max(pd_xy) - np.min(pd_xy))

        pd = np.concatenate([pd_xy,dof_3_channel,dof_4_channel],axis = -1)
        pd = np.nan_to_num(pd)

        assert pd.min() >= 0.0 and pd.max() <= 1.0
        assert len(pd.shape) == 3
        assert pd.shape[0] == self.number_of_xy_bins and pd.shape[1] == self.number_of_xy_bins and pd.shape[2] == (self.number_of_dofs - 2) * self.number_of_dof_bins + 1

        return pd

    def buildSM(self, channel_tensor, env_tensor, threshold): #, gazepath, lblpath, smpath):
        '''
        build saliency map and threshold it from motion plots of x,y in 1st channel
        '''
        single_channel = channel_tensor.reshape((224,224,1))*255
        rgb_image = np.concatenate([single_channel, single_channel, single_channel], axis=2)
        # single_channel_0 = np.zeros((224,224,1))
        # rgb_image =  np.concatenate([single_channel_0,single_channel,single_channel_0], axis=2)

        sm = SaliencyMap(rgb_image.astype('uint8'))
        gaze_tensor = (sm.map - sm.map.min()) / (sm.map.max() - sm.map.min())
        
        img = np.zeros((224,224,3))




        gaze_tensor = np.nan_to_num(gaze_tensor) # should be 3d array

        for i in range(224):
            for j in range(224):
                if gaze_tensor[i,j] > 0.05 and gaze_tensor[i,j] <= 0.40:
                    t1 = (gaze_tensor[i,j] - 0.05) / 0.35
                    img[i,j,:] = (t1,0,0)
                elif gaze_tensor[i,j] <= 0.80:
                    t1 = (gaze_tensor[i,j] - 0.40) / 0.40
                    img[i,j,:] = (0,t1,0)
                elif gaze_tensor[i,j] >= 0.80:
                    t1 = (gaze_tensor[i,j] - 0.80) / 0.20
                    img[i,j,:] = (0,0,t1) 
        img = img * 255.0
        gaze_image = gaze_tensor * 255
        
        # data in [0, 1]
        threshold_mask = np.ma.masked_where((env_tensor[:,:,0] == 0) | (gaze_tensor < threshold), gaze_image)
        gaze_image[threshold_mask.mask] = 0
        
        # data in [-1, 1]
        # threshold_mask = np.ma.masked_where((env_tensor[:,:,0] == -1) | (gaze_image <= threshold), gaze_image)
        # gaze_image[threshold_mask.mask] = -1

        gaze_image[~threshold_mask.mask] = 1

        return gaze_image
    
    def mergeLabel(self, channels_tensor, gaze_tensor):
        gaze_tensor = gaze_tensor.reshape((224, 224, 1))
        lbl = np.concatenate([gaze_tensor, channels_tensor[:,:,1:]], axis=2)
        assert lbl.shape[2] == 21

        # consider orientation of only those pixels that are identified as critical points by x,y (1st channel)
        # data in [0,1]
        mask = (lbl[:,:,0] == 0) 
        lbl[mask] = 0 

        print "gaze max: ",np.max(gaze_tensor)
        print "lbl max: ",np.max(lbl[:,:,1:])
        
        # data in [-1,1]
        # mask = (lbl[:,:,0] == -1) 
        # lbl[mask] = -1 

        # validate
        assert np.sum(np.isnan(lbl)) == 0
        assert len(lbl.shape) == 3 # 3D tensor
        assert lbl.shape[2] == 21 # should have 3 channels (0: critical region gaze, 1-2: dof values)
        for x in range(lbl.shape[2]): # all channels should be normalized
            assert lbl[:,:,x].min() >= 0.0 and lbl[:,:,x].max() <= 1.0

        lblpath = os.path.join(LBLDIR, str(self.envnum) + '.' + str(self.runnum) + '.npy')
        np.save(lblpath, lbl)
    
    def custom_sm(self,pd,threshold):
        mean = np.mean(pd)
        std = np.std(pd)
        # mask = ( pd < mean - (2 * std) ) or ( pd > mean + (2 * std))
        mask = np.ma.masked_where( ((pd < mean - (4 * std)) | (pd > mean + ( 4 * std))), pd )
        pd_copy = pd.copy()
        pd_copy[mask.mask] = 0
        m = np.max(pd_copy)
        new_pd = pd / m
        mask = np.ma.masked_where(new_pd < threshold,new_pd)
        new_pd[mask.mask] = 0
        new_pd[~mask.mask] = 1.0
        new_pd = self.smoothen_sm(new_pd)
        mask = np.ma.masked_where(new_pd < m,new_pd)
        new_pd[mask.mask] = 0
        new_pd[~mask.mask] = 1.0
        return new_pd

        
    def smoothen_sm(self,pd,n=2):
        return cv2.GaussianBlur(pd,(5,5),0)
 

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

        gaze_tensor = self.custom_sm(lbl[:,:,0],self.threshold)

        # gaze_tensor = self.buildSM(gaze_tensor,env_tensor,self.threshold)
        
        # gaze_tensor = lbl[:,:,0]
        self.mergeLabel(lbl,gaze_tensor)

if __name__ == '__main__':

    # print sys.argv
    envPath = sys.argv[1]
    envnum = float(sys.argv[2])
    runnum = int(sys.argv[3])
    visualize = False

    # envPath = 'envs/3d/2.0.dae'
    # envPath = './envs3d/17.1.xml'
    # envnum = 17.1
    # runnum = 1
    # visualize = False

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

    # DATADIR = os.path.join('data', 'env' + str(envnum), 'data')
    # INPDIR = os.path.join('data', 'env' + str(envnum), 'inp')
    # if not os.path.isdir(INPDIR):
    #     os.makedirs(INPDIR)
    # LBLDIR = os.path.join('data', 'env' + str(envnum), 'lbl')
    # if not os.path.isdir(LBLDIR):
    #     os.makedirs(LBLDIR)

    print('*********processing {envnum}.{runnum}*********'.format(envnum=str(envnum), runnum=str(runnum)))
    generator = LabelGenerator(env, envnum, runnum, bounds, 224)
    generator.generate()
