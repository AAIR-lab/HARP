from __future__ import division
from openravepy import *
from functools import partial
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import sys
from numpy import * 
import numpy
import numpy as np
import time
import os
from heapq import *
import pdb 
import cv2
import pickle
sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
from tmp.augment_pd_new import PDAugmenter

# import fetch

RaveSetDebugLevel(0)

# test_sample = sys.argv[1]
# NUMRUNS = int(sys.argv[2])
# MAXTIME = float(sys.argv[3])
# mode = sys.argv[4].lower()
# pd_mode = sys.argv[5].lower()
# guided = True if int(sys.argv[6]) == 1 else False
# constrained = True if int(sys.argv[7]) == 1 else False



# current = 0
# test_sample = "8.0.1"
test_sample = "10.2.1"
NUMRUNS = 10
MAXTIME = 1000000000
mode = "llp"
pd_mode = "learn"
guided = False
constrained = False
# 0

envnum = test_sample[:-2]
visualize = True
# visualize = False

if guided:
    gstr = "guided"
else:
    gstr = ""

if constrained:
    cstr = "constrained"
else:
    cstr = ""
seed = int(time.time())

results_file_plan = open(os.path.join('results','env' + envnum, envnum+'_time_' + mode + '_' + pd_mode + "_" + gstr + '.csv'), 'a')
results_file_time = open(os.path.join('results','env' + envnum, envnum+'_time_' + mode + '_' + pd_mode + "_" + gstr + '.csv'), 'a')
np.random.seed(seed)

def load_pd():
    pd = np.load(os.path.join('network','results', test_sample+'.npy'))
    # pd = pd.reshape((224,224,11))
    pd = np.squeeze(pd)
    xy_pd = pd[:,:,0]
    mask = np.ma.masked_where(xy_pd < 0.60, xy_pd)
    xy_pd[mask.mask] = 0.0
    xy_pd[~mask.mask] = 1.0
    pd[:,:,0] = xy_pd
    # pd = np.rot90(pd,k=3,axes=(0,1))

    return pd

# 843 823
# 841 821


class LL(object):

    def __init__(self,env,jointlimits,pd,n,m,s=[],g=[],pda = None, guided=False, constrained = False):
        super(LL, self).__init__()
        self.old = None
        self.selected_graph = 0
        self.constrained = constrained
        self.trace = []
        self.trace_2 = []
        self.samplecount = 0
        self.n = n
        self.m = m
        self.k = 0
        self.env = env
        self.pd = pd
        self.num_xy_bins = self.pd.shape[0]
        self.jointlimits = jointlimits
        self.image = None
        self.goalradius = 0.10
        self.currentgraph = 0
        self.currentstate = 'build graphs'
        self.s = s
        self.g = g
        # self.color = (0,1,0)
        self.color = (1,0,0)
        self.traceheight = 0.25
        self.number_of_bins = 20
        self.dof_bins = self.get_dof_bins()
        self.pda = pda
        s1 = self.convert_to_xy_bin(self.s)
        s2 = self.convert_to_xy_bin(self.g)
        path, flag = self.pda.check_path(s1, s2)
        if path is not None:
            new_pd = self.pda.update_pd(path)
            self.pd[:, :, 0] = new_pd
        if flag:
            self.path_found = True
            self.n = 250
        else:
            self.path_found = False
        self.guided = guided

        self.starttime = time.time()
        self.build_roadmap()

    def collides(self,q):
        collision = False
        robot = self.env.GetRobots()[0]	

        with self.env:
            prev = robot.GetActiveDOFValues()
            robot.SetActiveDOFValues(q)
            if self.env.CheckCollision(robot) or robot.CheckSelfCollision():
                collision = True
            robot.SetActiveDOFValues(prev)

        return collision

    def sample_uniform(self):		
        self.samplecount += 1
        state = [0.0 for i in xrange(len(self.jointlimits[0]))]
        for i, jlowlimit in enumerate(self.jointlimits[0]):
            state[i] = random.uniform(jlowlimit, self.jointlimits[1][i])

        # x_dof, y_dof = self.pda.generate_sample(pd = False)
        # dof_3 = np.random.choice(range(10))
        #
        # state = self.convert_sample((x_dof,y_dof,dof_3))

        return state

    def convert_sample(self, dof_sample):
        dof_values = []
        for i in range(len(dof_sample)):
            dof_value = (self.dof_bins[i]['bin_start'][dof_sample[i]] + self.dof_bins[i]['bin_end'][dof_sample[i]]) / 2.
            dof_values.append(dof_value)
        return dof_values

    def convert_to_xy_bin(self,sample):
        temp = []
        for i in range(2):
            value = sample[i]
            bins = self.dof_bins[i]
            if value < self.jointlimits[0][i]:
                temp.append(0)
            elif value > self.jointlimits[1][i]:
                temp.append(len(bins["bin_start"])-1)
            else:
                for j in range(len(bins["bin_start"])):
                    if bins["bin_start"][j] <= value < bins["bin_end"][j]:
                        temp.append(j)
                        break
        return temp


    # def sample_pd(self):
    #     dof_sample = []
    #     for i in range(self.pd.shape[0]):
    #         dof_bin = np.random.choice(range(self.number_of_bins), p=self.pd[i,:])
    #         dof_sample.append(dof_bin)

    #     dof_values = self.convert_sample(dof_sample)
    #     return dof_values

    def sample_pd(self):
        dof_sample = []
        xy_pd = self.pd[:,:,0].reshape((self.pd.shape[0] * self.pd.shape[1],))
        xy_pd = xy_pd / np.sum(xy_pd)
        xy_sample = np.random.choice(range(xy_pd.shape[0]),p = xy_pd)
        y_dof = int(xy_sample/self.pd.shape[1])
        x_dof = xy_sample - int(y_dof * self.pd.shape[1])

        # x_dof, y_dof = self.pda.generate_sample(pd=True)

        dof_3 = np.random.choice(range(10),p = self.pd[y_dof,x_dof,1:]/ np.sum( self.pd[y_dof,x_dof,1:]))
        # x_dof,y_dof = y_dof, x_dof
        # y_dof = -y_dof
        # x_dof = -x_dof

        # dof_3 = np.random.choice(range(10))
        # dof_4 = np.random.choice(range(10),p = self.pd[x_dof,y_dof,11:] / np.sum(self.pd[x_dof,y_dof,11:]) )

        dof_values = self.convert_sample((x_dof,y_dof,dof_3))
        # dof_values[0], dof_values[1] = dof_values[1], dof_values[0]
        # dof_values[1] = -dof_values[1]
        
        return dof_values

    def get_dof_bins(self):
        #create dof bins
        robot = self.env.GetRobots()[0]	
        llimits = robot.GetActiveDOFLimits()[0]
        ulimits = robot.GetActiveDOFLimits()[1]
        dof_ranges = ulimits - llimits
        number_of_xy_bins = self.num_xy_bins
        dof_bins = {}
        dof_bins = {}
        n_bin = 10
        # for x-dof
        i = 0
        dof_bins[i] = {}
        dof_bins[i]['bin_start'] = []
        dof_bins[i]['bin_end'] = []
        dof_bin_range = dof_ranges[i]/number_of_xy_bins
        s = llimits[i]
        for j in range(number_of_xy_bins):
            dof_bins[i]['bin_start'].append(s)
            dof_bins[i]['bin_end'].append(s + dof_bin_range)
            s += dof_bin_range
        # for y-dof
        i = 1
        dof_bins[i] = {}
        dof_bins[i]['bin_start'] = []
        dof_bins[i]['bin_end'] = []
        dof_bin_range = dof_ranges[i] / number_of_xy_bins
        s = ulimits[i]
        for j in range(number_of_xy_bins):
            dof_bins[i]['bin_start'].append(s - dof_bin_range)
            dof_bins[i]['bin_end'].append(s)
            s -= dof_bin_range

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
        return dof_bins

    def sample_samples(self):
        # rand = self.sample_uniform()
        randsample = np.random.choice(len(self.samples), size=1, replace=False)
        # minx, miny, maxx, maxy, [dof1, dof2]
        pointx = (self.samples[randsample[0]][0]+self.samples[randsample[0]][2])/2.0
        pointy = (self.samples[randsample[0]][1]+self.samples[randsample[0]][3])/2.0
        # state = rand[0:len(rand)-3]+[pointx,pointy]+[rand[len(rand)-1]]
        # randbin = np.random.choice(len(self.samples[randsample[0]][4]), size=1, replace=False)[0]
        # orientation_ = np.random.uniform(self.samples[randsample[0]][4][randbin][0], self.samples[randsample[0]][4][randbin][1])
        dof1 = self.samples[randsample[0]][4][0]
        dof2 = self.samples[randsample[0]][4][1]
        state = [pointx, pointy, dof1, dof2]
        # del self.samples[randsample[0]]
        return state

    def dist(self,p1,p2,flag=True):
        # if not self.constrained:
        if flag:
            a = numpy.array(p1)
            b = numpy.array(p2)
        else:
            a = numpy.array(p1[:-1])
            b = numpy.array(p2[:-1])

        return numpy.linalg.norm(a-b)

    def compound_step_constrained(self,p1,p2):
        if self.dist(p1,p2,False) <= 0.05:
            return p2
        else:
            # x = p1[0]
            # y = p1[1]
            # theta = p1[2]
            #
            # closest = None
            # closest_dist = float("inf")
            #
            # psi = -math.pi/6.0
            # while psi < math.pi/6.0:
            #     new_state = self.car_kinemetics(x,y,theta,psi,+1)
            #     dist = self.dist(p2,new_state)
            #     if dist < closest_dist:
            #         closest = new_state
            #         closest_dist = dist
            #     psi += math.pi/60.0
            dist = self.dist(p1,p2,False)
            dy = p1[1] - p2[1]
            dx = p1[0] - p2[0]
            # dy = p2[1] - p1[1]
            # dx = p2[0] - p1[0]
            theta = p1[2]
            psi = math.atan2(dy,dx)
            phi = psi - theta
            if phi > p1[2] + math.pi/4.0:
                phi = math.pi/4.0
            if phi < p1[2] - math.pi/4.0:
                phi = -math.pi/4.0



            dist = min(dist,0.1)
            new_x = p1[0] + dist * math.cos(theta)
            new_y = p1[1] + dist * math.sin(theta)
            new_theta = p1[2] + (dist/ 0.4) * math.tan(phi)

            return [new_x,new_y,new_theta]


            # psi = -math.pi/6.0
            # while psi < math.pi/6.0:
            #     new_state = self.car_kinemetics(x,y,theta,psi,-1)
            #     dist = self.dist(p2,new_state)
            #     if dist < closest_dist:
            #         closest = new_state
            #         closest_dist = dist
            #     psi += math.pi/60.0
                
            

        return closest
    
    def car_kinemetics(self,x,y,theta,psi,sign):
        delta_x = sign * 0.2 * math.cos(theta)
        # delta_x = sign * 0.2 * math.cos(theta+psi)
        # delta_y = sign * 0.2 * math.sin(theta+psi)
        delta_y = sign * 0.2 * math.sin(theta)
        delta_theta = sign * 0.2 / 0.4 * math.tan(psi)
        new_x = x + delta_x
        new_y = y + delta_y
        new_theta = theta + delta_theta
        return [new_x,new_y,new_theta]

    def compound_step(self,p1,p2):
        a = []
        for i in range(len(p1)):
            a = a + self.step_from_to([p1[i]],[p2[i]],0.1)

        return a

    def highlight_point(self,point):
        h = self.env.plot3(points=[point[0], point[1], 0.3], pointsize=0.07, colors=array([1,0,1]), drawstyle=1)
        return h

    def step_from_to(self,p1,p2,distance):
        #https://github.com/motion-planning/rrt-algorithms/blob/master/src/rrt/rrt_base.py
        if self.dist(p1,p2) <= distance:
            return p2
        else:
            a = numpy.array(p1)
            b = numpy.array(p2)
            ab = b-a  # difference between start and goal

            zero_vector = numpy.zeros(len(ab))

            ba_length = self.dist(zero_vector, ab)  # get length of vector ab
            unit_vector = numpy.fromiter((i / ba_length for i in ab), numpy.float, len(ab))
            # scale vector to desired length
            scaled_vector = numpy.fromiter((i * distance for i in unit_vector), numpy.float, len(unit_vector))
            steered_point = numpy.add(a, scaled_vector)  # add scaled vector to starting location for final point

            return list(steered_point)

    def goal_zone_collision(self,p1,p2):
        if self.dist(p1, p2) <= self.goalradius:
            return True
        else:
            return False

    def local_planner(self,s,g):
        ''' straight line planning '''
        result = False
        prev = s
        while True:
            step = self.compound_step(prev,g)
            if self.collides(step):
                break
            elif self.goal_zone_collision(step,g):
                result = True
                break
            else:
                prev = step 
        
        return result

    def connect(self, V, E, q):
        status = 'advanced'

        # loop until reached or collision
        while status is 'advanced':
            V, E, status, new = self.extend(V, E, q)

        if status == 'reached':
            # add G=(V,E) to q's graph
            i_q = len(self.roadmap[self.currentgraph])-1 
            self.roadmap[self.currentgraph] = self.roadmap[self.currentgraph] + V
            for i, e in enumerate(E):
                adj = []
                for n in E[e]:
                    adj.append((n[0]+i_q+1,n[1]))
                self.roadmapedges[self.currentgraph][i_q+1+i] = adj

            i_new = len(self.roadmap[self.currentgraph])-1
            self.roadmapedges[self.currentgraph][i_q].append((i_new,new))
            self.roadmapedges[self.currentgraph][i_new].append((i_q,q))
            # self.trace.append(self.env.drawlinestrip(points=array([[q[0], q[1], self.traceheight],[new[0], new[1], self.traceheight]]), linewidth=0.5, colors=array(self.color), drawstyle=1))		

        return status 

    
        
    def extend(self, V, E, q):
        i_near = distance.cdist([q], V).argmin()
        near = V[i_near]
        c1 = self.convert_to_xy_bin(near)
        c2 = self.convert_to_xy_bin(q)
        if self.guided and self.path_found:
            if not self.pda.is_neighbor(c1,c2) and not self.pda.is_in_same_state(c1,c2):
                return V,E,"trapped",None
        if self.constrained:
            new = self.compound_step_constrained(near,q)
        else:
            new = self.compound_step(near, q)
        if self.old == new:
            return V,E,"trapped",None
        else:
            self.old = new

        if self.collides(new) == False:  
            V.append(new)
            E[len(V)-1] = []
            E[len(V)-1].append((i_near,near)) 
            E[i_near].append((len(V)-1,new))
            # self.trace.append(self.env.plot3(points=[new[0], new[1], 0.25], pointsize=0.07, colors=array(self.color), drawstyle=1))
            self.trace.append(self.env.drawlinestrip(points=array([[near[0], near[1], 0.3],[new[0], new[1], 0.3]]), linewidth=2.0, colors=array(self.color), drawstyle=1))
            
            if self.goal_zone_collision(new, q):
                return V, E, 'reached', new
            else:
                return V, E, 'advanced', new
        else:
            return V, E, 'trapped', None 

    def connectN(self, q):
        self.env.plot3(points=[q[0],q[1],0.25],pointsize=0.07,colors=[0,1,1],drawstyle=1)
        delete = []
        for i in xrange(len(self.roadmap)):
            if i != self.currentgraph:
                connected = self.connect(self.roadmap[i], self.roadmapedges[i], q)
                if connected is 'reached': 
                    delete.append(i)
            if (time.time()-self.starttime) >= self.buildtime:
                break

        # delete merged graphs
        self.roadmap = [self.roadmap[i] for i in xrange(len(self.roadmap)) if not i in delete]
        self.roadmapedges = [self.roadmapedges[i] for i in xrange(len(self.roadmapedges)) if not i in delete]
        # print len(self.roadmap)
        # if self.constrained and len(self.roadmap) < 50:
        # if len(self.roadmap) < 20:
        if self.check_if_connected():
            return True


        return len(self.roadmap) == 1

    def check_if_connected(self):
        for i in range(len(self.roadmap)):
            if self.s in self.roadmap[i] and self.g in self.roadmap[i]:
                self.selected_graph = i
                return True
        return False


    def connect_2(self, V, E, q):
        status = 'advanced'

        # loop until reached or collision
        while status is 'advanced':
            V, E, status, new = self.extend(V, E, q)

        if status == 'reached':
            i_new = len(V)-1 
            V = V + self.roadmap[self.currentgraph]
            for i, e in enumerate(self.roadmapedges[self.currentgraph]):
                adj = []
                for n in self.roadmapedges[self.currentgraph][e]:
                    adj.append((n[0]+i_new-1,n[1]))
                E[i_new+1+i] = adj

            i_q = len(V)-1
            E[i_q].append((i_new,new))
            E[i_new].append((i_q,q))
            # self.trace.append(self.env.drawlinestrip(points=array([[q[0], q[1], self.traceheight],[new[0], new[1], self.traceheight]]), linewidth=0.5, colors=array(self.color), drawstyle=1))		

        return V, E, status 

    def connectN_2(self, q):
        merged = False

        if self.currentgraph == 0: 
            self.roadmap[1], self.roadmapedges[1], connected = self.connect_2(self.roadmap[1], self.roadmapedges[1], q)
            if connected is 'reached': 
                del self.roadmap[self.currentgraph]
                del self.roadmapedges[self.currentgraph]
                merged = True
                return True, merged
            else:
                return False, merged
        elif self.currentgraph == 1: 
            self.roadmap[0], self.roadmapedges[0], connected = self.connect_2(self.roadmap[0], self.roadmapedges[0], q)
            if connected is 'reached':
                del self.roadmap[self.currentgraph]
                del self.roadmapedges[self.currentgraph]
                merged = True
                return True, merged
            else:
                return False, merged
        else: 
            self.roadmap[0], self.roadmapedges[0], connected = self.connect_2(self.roadmap[0], self.roadmapedges[0], q)
            if connected is 'reached':
                del self.roadmap[self.currentgraph]
                del self.roadmapedges[self.currentgraph]
                merged = True
            else:
                self.roadmap[1], self.roadmapedges[1], connected = self.connect_2(self.roadmap[1], self.roadmapedges[1], q)
                if connected is 'reached':
                    del self.roadmap[self.currentgraph]
                    del self.roadmapedges[self.currentgraph]
                    merged = True
            return False, merged

    def swapN(self):
        if self.currentgraph >= len(self.roadmap)-1:
            self.currentgraph = 0
        else:
            self.currentgraph += 1

    def swapN_2(self, merged):
        if not merged:
            if self.currentgraph == len(self.roadmap)-1:
                self.currentgraph = 0
            else:
                self.currentgraph += 1
        else:
            # since merged trees, cluster number will already be set for new round; only change if merged tree at end of list
            if self.currentgraph >= len(self.roadmap):
                self.currentgraph = 0
    def plot(self,point,color):
        if self.image is None:
            self.image = np.zeros((224,224,3))

        x_bin = int(((point[0] - (-2.5)) / 5.0 ) * 224)
        y_bin = int(((point[1] - (-2.5)) / 5.0 ) * 224)
        
        for i in range(x_bin - 1,x_bin+2):
            for j in range(y_bin - 1 , y_bin +2 ):
                self.image[i,j,:] = color 
        
    def write_image(self):
        image = self.image * 255
        s = test_sample.split(".")
        inp_img = np.load("./test/env{}.{}/{}.npy".format(s[0],s[1],test_sample))[:,:,:3]
        inp_img = inp_img * 255

        inp_img = inp_img.astype(np.float32)
        image = image.astype(np.float32)

        final_img = cv2.addWeighted(image,0.8,inp_img,0.5,0)

        cv2.imwrite("./plots/{}_samples_{}_{}.png".format(test_sample,pd_mode,current),final_img)

    def get_next_sample(self):
        if self.k < 250:
            sample = self.sample_pd()
            self.k += 1
        else:
            sample = self.sample_uniform()
        # sample = self.sample_uniform()

        return sample
        

    def build_roadmap(self):
        global mode

        self.roadmap = []
        self.roadmapedges = []

        while len(self.roadmap) < self.n:
            while True:
                q = self.sample_pd()
                if not self.collides(q):
                    break
                else:
                    self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((0, 0, 1)), drawstyle=1)
            self.roadmap.append([q])
            self.roadmapedges.append({0:[]})
            self.trace.append(self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((1,0,0)), drawstyle=1))

        while (len(self.roadmap)-self.n) < self.m:
            while True:
                q = self.sample_uniform()
                if not self.collides(q):
                    break
            self.roadmap.append([q])
            self.roadmapedges.append({0:[]})
            # self.trace.append(self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((1,0,0)), drawstyle=1))

        # for i in self.roadmap:
        #     self.plot(i[0],[0,1,0])
        
        # self.plot(self.g,[1,1,1])
        # self.plot(self.s,[1,0,0])

        # self.write_image()

        # exit(0)

        if len(self.s) > 0 and len(self.g) > 0:
            self.roadmap.append([self.s])
            self.roadmapedges.append({0:[]})
            self.roadmap.append([self.g])
            self.roadmapedges.append({0:[]})

        if mode == 'llp':
            self.buildtime = MAXTIME
        else:
            self.buildtime = 2

        while True:
            if (time.time()-self.starttime) <= self.buildtime: # 1 seconds to build in RM mode. 60 in llp
                rand = self.get_next_sample()
                self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], status, new = self.extend(self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], rand)

                if status != 'trapped':
                    connected = self.connectN(new)

                    if connected:
                        # print str(time.time()-self.starttime) + ',' + str(len(self.roadmap[0])) # comment put for llp
                        self.currentstate = 'connected graphs'
                        break
                    pass

                if self.currentstate == 'build graphs':
                    self.swapN()
            else:
                print("Time up", str(time.time()-self.starttime))
                if mode == 'll-rm':
                    # comment out below in llp mode
                    self.currentstate = 'connected graphs'
                    numstates = 0
                    for m in self.roadmap:
                        numstates += len(m)
                    # print str(time.time()-self.starttime) + ',' + str(numstates)
                break

    def build_roadmap_prm(self):
        self.edges = {}
        self.vertices = []
        self.k = 3

        while len(self.vertices) < self.n:
            while True:
                q = self.sample_samples()
                if not self.collides(q):
                    break
            self.vertices.append(q)
            # self.tracfloatppend(self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((1,0,0)), drawstyle=1))

        while (len(self.vertices)-self.n) < self.m:
            while True:
                q = self.sample_uniform()
                if not self.collides(q):
                    break
            self.vertices.append(q)
            # self.trace.append(self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((0,0,1)), drawstyle=1))

        self.kdtree = KDTree(self.vertices)
         
        for i_v, q in enumerate(self.vertices):
            dist, ind = self.kdtree.query([q],k=self.k+1)
            self.edges[i_v] = []
            for i_n in ind[0]:
                if i_n != i_v:
                    n = self.vertices[i_n]
                    if (not (i_n,n) in self.edges[i_v]) and self.local_planner(q,n):
                        self.edges[i_v].append((i_n,n))
                        # self.trace.append(self.env.drawlinestrip(points=array([[q[0], q[1], 0.25],[n[0], n[1], 0.25]]), linewidth=0.5, colors=array((0,1,0)), drawstyle=1))

        return True

    def search(self):
        ''' dijkstra's '''
        q = []
        dist = {}
        prev = {}
        
        for i in xrange(len(self.roadmap[self.selected_graph])):
            dist[i] = inf
            prev[i] = None

        dist[self.start[0]] = 0
        heappush(q, (0,self.start))

        while q:
            currdist, near = heappop(q)

            for n in self.roadmapedges[self.selected_graph][near[0]]:
                alt = currdist + self.dist(near[1], n[1])
                if alt < dist[n[0]]:
                    dist[n[0]] = alt
                    prev[n[0]] = near
                    heappush(q, (alt, n))

        # collect solution path through backtracking from goal using prev
        solutiontrace = []
        temp = self.goal
        if prev[temp[0]]:
            while temp:
                solutiontrace.append(temp[1])
                temp = prev[temp[0]]

        return solutiontrace

    def refine(self,path):
        solution = []
        pdb.set_trace()
        for i in [0, len(path)-2]:
            section = []
            prev = path[i]
            section.append(prev)
            # self.trace.append(self.env.plot3(points=[prev[0], prev[1], 0.25], pointsize=0.03, colors=array((0,1,0)), drawstyle=1))
            while not self.goal_zone_collision(step, path[i+1]):
                step = self.compound_step(prev,path[i+1])
                section.append(step)
                # self.trace.append(self.env.plot3(points=[step[0], step[1], 0.25], pointsize=0.03, colors=array((0,1,0)), drawstyle=1))
                # self.trace.append(self.env.drawlinestrip(points=array([[prev[0], prev[1], 0.25],[step[i+1][0], step[i+1][1], 0.25]]), linewidth=0.5, colors=array((0,1,0)), drawstyle=1))
                prev = step
            solution.append(section)
        #pdb.set_trace()
        return solution

    def update_pda(self,plan):
        binned_trajectory = []
        for point in plan:
            binned_point = self.convert_to_xy_bin(point)
            binned_trajectory.append(binned_point)

        start = self.convert_to_xy_bin(self.s)
        goal = self.convert_to_xy_bin(self.g)
        self.pda.add_exp(binned_trajectory,start,goal)
        return self.pda, binned_trajectory

    def visualize(self, solutiontrace):
        for h in self.trace:
            h.Close()
        robot = self.env.GetRobots()[0]
        tracerobots = []
        
        for point in solutiontrace:
            robot.SetActiveDOFValues(point)
            time.sleep(0.5)
        
        '''
        time.sleep(0.1)
        for point in solutiontrace:
            robot.SetActiveDOFValues(point)
            newrobot = RaveCreateRobot(self.env,robot.GetXMLId())
            newrobot.Clone(robot,0)
            #for link in newrobot.GetLinks():
            #	for geom in link.GetGeometries():
            #		geom.SetTransparency(0.5)	
            newrobot.SetTransform(robot.GetTransform())
            self.env.Add(newrobot,True)
            tracerobots.append(newrobot)
            time.sleep(0.025)		

        self.env.UpdatePublishedBodies()
        pdb.set_trace()
        for bot in tracerobots:
            self.env.Remove(bot)
            time.sleep(0.01)
        del tracerobots
        '''
        #pdb.set_trace()

    def plan(self,s,g,visualize=False):
        global mode

        if mode == 'll-rm':
            self.starttime = time.time() # comment out for llp

        if len(self.s) == 0 and len(self.g) == 0:
            self.currentgraph = 0
            self.currentstate = 'build graphs'
            # self.color = (1,0,1)
            self.traceheight = 0.3
            self.roadmap.append([s])
            self.roadmapedges.append({0:[]})
            self.roadmap.append([g])
            self.roadmapedges.append({0:[]})
            self.buildtime = MAXTIME
            while (time.time()-self.starttime) <= MAXTIME:
                rand = self.get_next_sample()
                self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], status, new = self.extend(self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], rand)
                if status != 'trapped':
                    connected = self.connectN(new)

                    if connected:
                        self.currentstate = 'connected graphs' 
                        break

                if self.currentstate == 'build graphs':
                    self.swapN()

        if self.currentstate == 'connected graphs':
            i_s = distance.cdist([s], self.roadmap[self.selected_graph]).argmin()
            i_g = distance.cdist([g], self.roadmap[self.selected_graph]).argmin()
            self.start = (i_s,s)
            self.goal = (i_g,g)

            path = self.search()
            roadmap_size = 0
            for i in self.roadmap:
                roadmap_size += len(i)
            if len(path) > 0:
                print "Path length: ", len(path)
                print "Roadmap size: ", roadmap_size
                print "Completed in " + str(time.time()-self.starttime)
                path = path[::-1]
                if visualize:
                    # path = path[::-1]
                    prev = None
                    for point in path:
                        # self.trace_2.append(self.env.plot3(points=[point[0], point[1], 0.3], pointsize=0.05, colors=array((1,1,1)), drawstyle=1))
                        if prev is not None:
                            self.trace.append(self.env.drawlinestrip(points=array([[prev[0], prev[1], 0.3],[point[0],point[1] ,0.5]]), linewidth=2.0, colors=array((1,1,1)), drawstyle=1))
                        prev = point
                    time.sleep(0.1)

                    # self.visualize(path)
                del self.trace
                self.trace = []
                return True, path
            else:
                del self.trace
                self.trace = []
                return False, []
        else:
            numstates = 0
            for m in self.roadmap:
                numstates += len(m)
            self.image = None
            print 'out of time' + ',' + str(time.time()-self.starttime) + ',' + str(numstates)
            del self.trace
            self.trace = []
            return False, []

def setupRobot(robot):
    if robot.GetName() == 'robot1' or robot.GetName() == "bot":
        robot.SetDOFVelocities([0.5,0.5,0.5])
        robot.SetActiveDOFs([0,1,2])
        robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
    elif robot.GetName() == 'lshaped':
        robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
        robot.SetActiveDOFs([0,1,2,4])
        robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])
    elif robot.GetName() == 'fetch':
        llimits = robot.GetDOFLimits()[0]
        ulimits = robot.GetDOFLimits()[1]
        # DOFs 2, 12, 14 are circular and need their limits to be set to [-3.14, 3.14] as stated by planner
        llimits[2] = llimits[12] = llimits[14] = -np.pi
        ulimits[2] = ulimits[12] = ulimits[14] = np.pi
        robot.SetDOFLimits(llimits, ulimits)

def load_inp(envnum,test_sample):
    inp = np.squeeze(np.load(os.path.join('test','env'+str(envnum), str(test_sample)+'.npy')))
    return inp

def main():

    global mode
    global current

    env = Environment()
    env.Load('envs3d/'+envnum+'.xml')
    inp = load_inp(envnum,test_sample)

    
    if visualize:
        env.SetViewer('qtcoin')
        viewer = env.GetViewer()
        viewer.SetSize(900,900)
        viewer.SetCamera([[  9.99989147e-01,  -3.08830989e-03,   3.48815543e-03,
		          3.30109894e-02],
		       [ -3.08664267e-03,  -9.99995120e-01,  -4.83250519e-04,
		          5.10825627e-02],
		       [  3.48963084e-03,   4.72478585e-04,  -9.99993800e-01,
		          6.83978987e+00],
		       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
		          1.00000000e+00]])

    
    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    collisionChecker.SetCollisionOptions(CollisionOptions.Contacts)
    env.SetCollisionChecker(collisionChecker)
    

    # load robots
    robot = env.GetRobots()[0]
    setupRobot(robot)
    env.UpdatePublishedBodies()

    jointlimits = robot.GetActiveDOFLimits()

    # tuck
    
    if test_sample in ['10.2.1', '10.2.2', '10.0.1', '10.0.2']:
            start = [0.5443226027230188, 0.9758539009870815, -1.7579191386448807]
        # 8.0
    elif test_sample in ['8.0.1', '8.0.4']:
        start = [-1.6, -1.0, 0]
    elif test_sample in ['8.0.2', '8.0.3']:
        start = [1.6, 1.0, 0]

    goal_file = open("test/env{}/goal_{}.pkl".format(envnum,test_sample),"rb")
    goal = pickle.load(goal_file)
    with env:
        robot.SetActiveDOFValues(goal)
        if env.CheckCollision(robot) or robot.CheckSelfCollision():
            print "Goal in collision"
            exit(-1)
        robot.SetActiveDOFValues(start)
        if env.CheckCollision(robot) or robot.CheckSelfCollision():
            print "Init in collision"
            exit(-1)    
    # robot.SetActiveDOFValues(goal)
    results_file = open(
        os.path.join('results', 'env' + envnum,  str(seed) + "_" + str(start) + "_" + str(goal) + '.csv'), 'a')
    robot.SetActiveDOFValues(start)
    print("start: ", start)
    print("goal: ", goal)
    if visualize:
        raw_input("start..?")

    pd = load_pd()


    # start_time = time.time()
    # if guided: 
    #     try:
    #         pda = pickle.load(open("pda_{}_constrained.p".format(test_sample),"rb"))
    #     except:
    #         pda = PDAugmenter(pd[:,:,0],0.75,inp,0.4)
    #         # pickle.dump(pda,open("pda_{}_constrained.p".format(test_sample),"wb"))
    # print "Time to build the graph: {}".format(time.time() - start_time)


    pda = PDAugmenter(pd[:,:,0],0.6,inp,0.4)
    pda.plot()
    if pd_mode == "learn":
        m = 0
        n = 500
    elif pd_mode == "random":
        m = 250
        n = 0 
    if mode == 'llp':
        total_success = 0
        for i in xrange(NUMRUNS):
            current  = i
            problem = LL(env,jointlimits,pd,n,m,start,goal,pda=pda,guided=guided,constrained=constrained)
            starttime = time.time()
            success, path = problem.plan(start,goal, visualize)
            if success:
                total_success += 1
            print success, total_success, i+1
            print 'LLP', MAXTIME
            print '*********'
            results_file_time.write(str(i) + ',' + str(success) + ',' + str(time.time() - problem.starttime) + '\n')
            pda,binned_path = problem.update_pda(path)
            pda.plot("temp_{}".format(i),mp=binned_path)
            print 'Completed.'
        print test_sample, 'LLP', str(MAXTIME), total_success
        results_file.write(str(MAXTIME) + ',' + str(total_success) + '\n')
    else:
        m = 1000
        n = 250
        total_success = 0
        for i in xrange(NUMRUNS):
            problem = LL(env,jointlimits,pd,n,m, pda=pda, guided = guided)
            starttime = time.time()
            success, path = problem.plan(start,goal,visualize)
            if success:
                total_success += 1
            print success, total_success, i+1
            print 'LL-RM', MAXTIME
            print '*********'
        print 'Completed.'
        print test_sample, 'LL-RM', str(MAXTIME), total_success
        results_file.write(str(MAXTIME) + ',' + str(total_success) + '\n')

        results_file.close()
        RaveDestroy()

if __name__ == "__main__":
    main()
