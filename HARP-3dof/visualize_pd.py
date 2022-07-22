from openravepy import *
import numpy as np
import cv2
import copy
import sys


class PDVisualizer(object):

    def __init__(self,pd,envpath,env,envnum,problem_numnber,datainp,label):
        self.pd = pd
        self.number_of_bins = 10
        self.env = env
        self.robot = self.env.GetRobots()[0]
        self.setupRobot()
        self.dof_bins = self.get_dof_bins()
        self.envpath = envpath
        self.envnum = envnum
        self.problem_number = problem_number
        self.inp = np.load(datainp)
        self.make_image()

    def setupRobot(self):
        if self.robot.GetName() == 'robot1' or self.robot.GetName() == 'bot':
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

    
    def convert_sample(self, dof_sample):
        dof_values = []
        for i in range(len(dof_sample)):
            # dof_value = (self.dof_bins[i]['bin_start'][dof_sample[i]] + self.dof_bins[i]['bin_end'][dof_sample[i]]) / 2.
            dof_value = np.random.uniform(self.dof_bins[i]['bin_start'][dof_sample[i]], self.dof_bins[i]['bin_end'][dof_sample[i]] )
            dof_values.append(dof_value)
        return dof_values


    def sample_pd(self):
        dof_sample = []
        for i in range(self.pd.shape[0]):
            dof_bin = np.random.choice(range(self.number_of_bins), p=self.pd[i,:])
            dof_sample.append(dof_bin)

        dof_values = self.convert_sample(dof_sample)
        return dof_values

    def get_dof_bins(self):
        #create dof bins
        robot = self.env.GetRobots()[0]	
        llimits = robot.GetActiveDOFLimits()[0]
        ulimits = robot.GetActiveDOFLimits()[1]
        dof_ranges = ulimits - llimits
        dof_bins = {}
        for i in range(len(dof_ranges)):
            dof_bins[i] = {}
            dof_bins[i]['bin_start'] = []
            dof_bins[i]['bin_end'] = []
            dof_bin_range = dof_ranges[i]/self.number_of_bins
            s = llimits[i]
            for j in range(self.number_of_bins):
                dof_bins[i]['bin_start'].append(s)
                dof_bins[i]['bin_end'].append(s + dof_bin_range)
                s += dof_bin_range
        return dof_bins

    def get_color(self,dof_value,dof_number):
        x = (dof_value - self.dof_bins[dof_number]["bin_start"][0]) / ( self.dof_bins[dof_number]["bin_end"][-1] -  self.dof_bins[dof_number]["bin_start"][0])
        # return abs((x*2)-1)
        return x
        # return tuple(new_color)

    def make_image(self):
        inp = self.inp
        inp = inp[:,:,0]
        inp = inp.astype(np.float32)
        cp = inp.copy()
        pd = self.pd 
        xy_pd = pd[:,:,0]
        np.savetxt("./plots/{}.csv".format(self.problem_number),xy_pd,delimiter=",")
        xy_pd_1 = xy_pd.reshape((224,224,1))
        xy_pd_3 = np.concatenate([xy_pd_1,xy_pd_1,xy_pd_1],axis = -1)
        xy_image = xy_pd_3 * 255.0
        xy_image = xy_image.astype(np.float32)

        dof_image = np.zeros(xy_image.shape)

        for i in range(dof_image.shape[0]):
            for j in range(dof_image.shape[1]):
                if self.pd.shape[-1] == 2:
                    max_bin = pd[i,j,1]
                else:
                    max_bin = np.argmax(pd[i,j,1:])
                # max_bin = abs(5 - max_bin) / float(5)
                if max_bin in [1,2,7,8]:
                    dof_image[i,j,:] = (0,1,0) #robot is horizontal
                else:
                    dof_image[i,j,:] = (1,0,0)  #robot is vertical
                # dof_image[i,j,:] = (max_bin,0,0)

        img_mask = np.ma.masked_where(inp==0,inp)

        xy_image = np.zeros((224,224))
        xy_image[~img_mask.mask] = 1.0
        xy_image_green_channel = xy_image.copy()
        xy_image_red_channel = xy_image.copy()


        mask = xy_pd < 0.6
        mask2 = xy_pd >= 0.6

        xy_image_red_channel[mask2] = 1.0
        xy_image[mask2] = 0.0
        xy_image_green_channel[mask2] = 0.0

        xy_image = xy_image.reshape((224,224,1))
        xy_image_green_channel = xy_image_green_channel.reshape((224,224,1))
        xy_image_red_channel = xy_image_red_channel.reshape((224,224,1))
        xy_image = np.concatenate([xy_image,xy_image_green_channel,xy_image_red_channel],axis = -1)

        dof_image_final = np.zeros(xy_image.shape)

        # for i in range(224):
        #     for j in range(224):
                

        mask = xy_image[:,:,0] == 1.0
        mask2 = inp == 0

        dof_image_0 = dof_image[:,:,0]
        dof_image_1 = dof_image[:,:,1]
        dof_image_2 = dof_image[:,:,2]

        dof_image_0[mask] = 1.0
        dof_image_0[mask2] = 0.0
        dof_image_1[mask] = 1.0
        dof_image_1[mask2] = 0.0
        dof_image_2[mask] = 1.0
        dof_image_2[mask2] = 0.0

        dof_image_0 = dof_image_0.reshape((224,224,1))
        dof_image_1 = dof_image_1.reshape((224,224,1))
        dof_image_2 = dof_image_2.reshape((224,224,1))

        dof_image = np.concatenate([dof_image_0,dof_image_1,dof_image_2],axis = -1)

        dof_image = dof_image * 255.0
        xy_image = xy_image * 255.0

        # inp = inp * 255.0
        # final_img = cv2.addWeighted(xy_image,1,inp,1,0)
        # final_img = cv2.addWeighted(inp,1,xy_image,1,0)
        # final_img2 = cv2.addWeighted(dof_image,1,inp,1,0)
        cv2.imwrite("./plots/env_{}.png".format(self.problem_number), cp * 255)
        cv2.imwrite("./plots/{}.png".format(self.problem_number),xy_image)
        cv2.imwrite("./plots/{}_dof.png".format(self.problem_number),dof_image)

if __name__ == "__main__":

    # envNum = "17.1"
    # runnum = "49"
    # envtype = "train"
    # label = "train"

    envNum = sys.argv[1]
    runnum = sys.argv[2]
    envtype = sys.argv[3]
    label = sys.argv[4]

    problem_number = envNum + "." + runnum
    envPath = "envs3d/{}.xml".format(envNum)

    env = Environment()
    env.Load(envPath)

    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    collisionChecker.SetCollisionOptions(CollisionOptions.Contacts)
    env.SetCollisionChecker(collisionChecker)

    # pd = np.load("network/results/{}.npy".format(problem_number)).reshape((4,10))

    if envtype == "train":
        datainp = "data/env{}/inp/{}.npy".format(envNum,problem_number)
    else:
        datainp = "test/env{}/{}.npy".format(envNum,problem_number)

    if label == "test":
        pd = np.load("network/results/{}.npy".format(problem_number))[0,:,:,:]
    else:
        pd = np.load("data/env{}/lbl/{}.npy".format(envNum,problem_number))

    PDVisualizer(pd,envPath,env,envNum,problem_number,datainp,label)
