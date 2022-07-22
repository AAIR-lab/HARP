from openravepy import *
import numpy as np
import cv2
import copy
import sys
import os


def load_pd(test_sample):
    final_pd =  None 
    pd = None
    for i in range(5):
        for j in range(5):
            if pd is None:
                pd = np.squeeze(np.load(os.path.join('network','results', str(test_sample) + "_" + str(j) + "_" + str(i) +   '.npy')) )
            else:
                t = np.squeeze(np.load(os.path.join('network','results', str(test_sample) + "_" + str(j) + "_" + str(i) +   '.npy')))
                pd = np.hstack((pd,t))
        if final_pd is None:
            final_pd = pd 
            pd = None
        else:
            final_pd = np.vstack((pd,final_pd))
            pd = None

    pd = final_pd.copy()

    # pd = np.load(os.path.join('network','results', test_sample+'.npy'))
    # # pd = pd.reshape((224,224,11))
    # pd = np.squeeze(pd)
    # xy_pd = pd[:,:,0]
    # mask = np.ma.masked_where(xy_pd < 0.60, xy_pd)
    # xy_pd[mask.mask] = 0.0
    # xy_pd[~mask.mask] = 1.0
    # pd[:,:,0] = xy_pd
    return pd

def load_inp(envnum, test_sample):
    final_pd =  None 
    pd = None
    for i in range(5):
        for j in range(5):
            if pd is None:
                pd = np.squeeze(np.load(os.path.join('test','env'+str(envnum) , str(test_sample) + "_" + str(j) + "_" + str(i) +   '.npy')))
            else:
                t = np.squeeze(np.load(os.path.join('test','env'+str(envnum), str(test_sample) + "_" + str(j) + "_" + str(i) +   '.npy')))
                pd = np.hstack((pd,t))
        if final_pd is None:
            final_pd = pd 
            pd = None
        else:
            final_pd = np.vstack((pd,final_pd))
            pd = None

    pd = final_pd.copy()
    return pd



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
        self.inp = datainp
        self.make_image()

    def setupRobot(self):
        if self.robot.GetName() == 'robot1' or self.robot.GetName() == 'bot':
            self.robot.SetDOFVelocities([0.5,0.5,0.5])
            self.robot.SetActiveDOFs([0,1,2])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
        elif self.robot.GetName() == 'lshaped':
            self.robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
            self.robot.SetActiveDOFs([0,1,2,3])
            self.robot.SetDOFLimits([-2.5, -2.5, -np.pi, -np.pi/2], [2.5, 2.5, np.pi, np.pi/2])
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
        inp = inp[:, :, :3]
        inp = inp.astype(np.float32)
        pd = self.pd
        xy_pd = pd[:, :, 0]
        np.savetxt("./plots/{}.csv".format(self.problem_number), xy_pd, delimiter=",")
        xy_pd_1 = xy_pd.reshape((inp.shape[0], inp.shape[0], 1))
        xy_pd_3 = np.concatenate([xy_pd_1, xy_pd_1, xy_pd_1], axis=-1)
        xy_image = xy_pd_3 * 255.0
        xy_image = xy_image.astype(np.float32)

        dof_image = np.zeros(xy_image.shape)
        dof_image2 = np.zeros(xy_image.shape)

        for i in range(dof_image.shape[0]):
            for j in range(dof_image.shape[1]):
                if self.pd.shape[-1] == 3:
                    max_bin = pd[i, j, 1]
                    max_bin2 = pd[i, j, 2]
                else:
                    max_bin = np.argmax(pd[i, j, 1:11])
                    max_bin2 = np.argmax(pd[i, j, 11:])

                # max_bin = abs(5 - max_bin) / float(5)
                # max_bin2 = abs(5 - max_bin) / float(5)

                if max_bin in [1, 2, 7, 8]:
                    dof_image[i, j, :] = (0, 1, 0)
                else:
                    dof_image[i, j, :] = (1, 0, 0)

                if max_bin2 in [4, 5, 6]:
                    dof_image2[i, j, :] = (1, 0, 0)
                else:
                    dof_image2[i, j, :] = (0, 1, 0)

        mask = xy_pd < 0.6
        mask2 = xy_pd >= 0.6
        dof_image[mask] = 0
        dof_image2[mask] = 0

        xy_image[mask] = 0
        xy_image[mask2] = 255.0
        dof_image = dof_image * 255.0
        dof_image = dof_image.astype(np.float32)

        dof_image2 = dof_image2 * 255.0
        dof_image2 = dof_image2.astype(np.float32)

        inp = inp * 255.0
        final_img = cv2.addWeighted(xy_image, 1.0, inp, 0.5, 0)
        # final_img2 = cv2.addWeighted(dof_image,1.0,inp,0.5,0)
        # final_img3 = cv2.addWeighted(dof_image2,1.0,inp,0.5,0)

        final_img2 = np.zeros(final_img.shape)
        final_img3 = np.zeros(final_img.shape)

        mask = final_img[:, :, 0] > 255.0 * 0.7
        mask2 = final_img[:, :, 0] == 0

        final_img = np.ones((inp.shape[0], inp.shape[0]))
        final_img[mask] = 0
        final_img[mask2] = 0
        final_img = final_img.reshape((inp.shape[0], inp.shape[0], 1))
        final_img = np.concatenate([final_img, final_img, final_img], axis=-1)

        temp = np.ones((inp.shape[0], inp.shape[0]))
        temp[mask] = 1
        temp[mask2] = 0
        final_img[:, :, 2] = temp

        for i in range(inp.shape[0]):
            for j in range(inp.shape[0]):
                if tuple(final_img[i, j, :]) == (1, 1, 1):
                    final_img2[i, j, :] = (1, 1, 1)
                    final_img3[i, j, :] = (1, 1, 1)
                else:
                    if tuple(final_img[i, j, :]) == (0, 0, 1):
                        final_img2[i, j, :] = dof_image[i, j, :]
                        final_img3[i, j, :] = dof_image2[i, j, :]

        final_img = final_img * 255.0
        final_img2 = final_img2 * 255.0
        final_img3 = final_img3 * 255.0
        cv2.imwrite("./plots/{}.png".format(self.problem_number), final_img)
        cv2.imwrite("./plots/{}_dof.png".format(self.problem_number), final_img2)
        cv2.imwrite("./plots/{}_dof2.png".format(self.problem_number), final_img3)

if __name__ == "__main__":
    #
    # envNum = "32.0"
    # runnum = "1"
    # envtype = "test"
    # label = "test"

    envNum = sys.argv[1]
    runnum = sys.argv[2]
    envtype = sys.argv[3]
    label = sys.argv[4]

    problem_number = envNum + "." + runnum
    # envPath = "envs3d/{}.xml".format(envNum)
    envPath = "envs3d/{}.dae".format(envNum)

    env = Environment()
    env.Load(envPath)

    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    collisionChecker.SetCollisionOptions(CollisionOptions.Contacts)
    env.SetCollisionChecker(collisionChecker)

    # pd = np.load("network/results/{}.npy".format(problem_number)).reshape((4,10))

    if envtype == "train":
        datainp = "data/env{}/inp/{}.npy".format(envNum,problem_number)
    else:
        datainp = load_inp(envNum,problem_number)

    if label == "test":
        pd = load_pd(problem_number)
    else:
        pd = np.load("data/env{}/lbl/{}.npy".format(envNum,problem_number))

    PDVisualizer(pd,envPath,env,envNum,problem_number,datainp,label)
