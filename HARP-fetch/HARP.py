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
import pickle
sys.path.append(os.path.join('envs', '3d', 'robots', 'fetch'))
import fetch

RaveSetDebugLevel(0)

test_sample = sys.argv[1]
NUMRUNS = int(sys.argv[2])
MAXTIME = float(sys.argv[3])
mode = sys.argv[4].lower()

# test_sample = '7.3.1'
# NUMRUNS = 100
# MAXTIME = 20.0
# mode = 'llp'

envnum = test_sample[:-2]
visualize = False

results_file = open(os.path.join('results', 'env' + test_sample, 'plan_' + mode + '.csv'), 'a')
np.random.seed(None) #random seed

def load_pd():
    pd = np.load(os.path.join('network','results', test_sample+'.npy'))
    # pd = np.load("./data-v2/env1.0/lbl/1.0.1.npy")
    if pd.shape[-1] == 9:
        pd = pd.reshape((64,64,64,9))
    else:
        pd = pd.reshape((64,64,64,81))
    xyz_pd = pd[:,:,:,0]
    flag = xyz_pd < 0.35
    xyz_pd[flag] = 0.0
    xyz_pd[~flag] = 1.0
    pd[:,:,:,0] = xyz_pd
    return pd

class LL(object):

    def __init__(self,env,jointlimits,pd,n,m,s=[],g=[]):
        super(LL, self).__init__()
        self.starttime = time.time()
        self.trace = []
        self.robot = env.GetRobots()[0]
        self.bounds = [[-1, -1, 0], [1, 1, 2]]
        self.samplecount = 0
        self.n = n
        self.m = m
        self.env = env
        self.pd = pd
        self.jointlimits = jointlimits
        self.goalradius = 0.25
        
        self.number_of_dof_bins = 10
        self.labelsize = 64
        self.currentgraph = 0
        self.currentstate = 'build graphs'
        self.s = s
        self.g = g
        self.color = (0,1,0)
        self.traceheight = 0.25
        self.number_of_bins = 10
        self.xyz_bins, self.dof_bins = self.get_dof_bins()
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

        return state

    def convert_sample(self, dof_sample):
        dof_values = []
        for i in range(len(dof_sample)):
            dof_value = (self.dof_bins[i]['bin_start'][dof_sample[i]] + self.dof_bins[i]['bin_end'][dof_sample[i]]) / 2.
            dof_values.append(dof_value)
        return dof_values

    def sample_pd(self):
        dof_sample = []
        xyz_pd = self.pd[:,:,:,0]
        xyz_flat = xyz_pd.reshape((64*64*64,))
        xyz_sample = np.random.choice(range(xyz_flat.shape[0]),p=xyz_flat/np.sum(xyz_flat))
        x_dof = int(xyz_sample/(64*64))
        yz_sample = xyz_sample - int(x_dof * 64 * 64)
        y_dof = int(yz_sample/64)
        z_dof = yz_sample - int(y_dof * 64)


        for i in range(8):
            if self.pd.shape[-1] == 9:
                dof = self.pd[x_dof,y_dof,z_dof,i]
            else:
                dof = np.random.choice(range(10),p = self.pd[x_dof,y_dof,z_dof,i*10+1:(i*10+1)+10])
            dof_sample.append(int(dof))

        dof_values = self.convert_sample(dof_sample)
        return dof_values

    def get_dof_bins(self):
        #create dof bins
        n_bin = self.number_of_dof_bins
        llimits = self.robot.GetActiveDOFLimits()[0]
        ulimits = self.robot.GetActiveDOFLimits()[1]
        dof_ranges = ulimits - llimits
        dof_bins = {}
        xyz_bins = {}
        for i in range(3):
            step = (self.bounds[1][i] - self.bounds[0][i]) / float(self.labelsize)
            xyz_bins[i] = {}
            xyz_bins[i]['bin_start'] = []
            xyz_bins[i]['bin_end'] = []
            s = self.bounds[0][i]
            for j in range(self.labelsize):
                xyz_bins[i]['bin_start'].append(s)
                xyz_bins[i]['bin_end'].append(s+step)
                s += step
        
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

        return xyz_bins, dof_bins

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

    def dist(self,p1,p2):
        a = numpy.array(p1)
        b = numpy.array(p2)

        return numpy.linalg.norm(a-b)

    def compound_step(self,p1,p2):
        a = []
        for i in range(len(p1)):
            a = a + self.step_from_to([p1[i]],[p2[i]],0.2)

        return a

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
        new = self.compound_step(near, q)

        if self.collides(new) == False:  
            V.append(new)
            E[len(V)-1] = []
            E[len(V)-1].append((i_near,near)) 
            E[i_near].append((len(V)-1,new))
            # self.trace.append(self.env.plot3(points=[new[0], new[1], self.traceheight], pointsize=0.03, colors=array(self.color), drawstyle=1))
            # self.trace.append(self.env.drawlinestrip(points=array([[near[0], near[1], self.traceheight],[new[0], new[1], 0.25]]), linewidth=0.5, colors=array(self.color), drawstyle=1))
            
            if self.goal_zone_collision(new, q):
                return V, E, 'reached', new
            else:
                return V, E, 'advanced', new
        else:
            return V, E, 'trapped', None 

    def connectN(self, q):
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

        return len(self.roadmap) == 1

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

    def build_roadmap(self):
        global mode

        self.roadmap = []
        self.roadmapedges = []

        while len(self.roadmap) < self.n:
            while True:
                q = self.sample_pd()
                if not self.collides(q):
                    break
            self.roadmap.append([q])
            self.roadmapedges.append({0:[]})
            # self.trace.append(self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((1,0,0)), drawstyle=1))

        while (len(self.roadmap)-self.n) < self.m:
            while True:
                q = self.sample_uniform()
                if not self.collides(q):
                    break
            self.roadmap.append([q])
            self.roadmapedges.append({0:[]})
            # self.trace.append(self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((0,0,1)), drawstyle=1))

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
                rand = self.sample_uniform()
                self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], status, new = self.extend(self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], rand)

                if status != 'trapped':
                    connected = self.connectN(new)

                    if connected:
                        # print str(time.time()-self.starttime) + ',' + str(len(self.roadmap[0])) # comment put for llp
                        self.currentstate = 'connected graphs'
                        break

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
            # self.trace.append(self.env.plot3(points=[q[0], q[1], 0.25], pointsize=0.03, colors=array((1,0,0)), drawstyle=1))

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
        
        for i in xrange(len(self.roadmap[0])):
            dist[i] = inf
            prev[i] = None

        dist[self.start[0]] = 0
        heappush(q, (0,self.start))

        while q:
            currdist, near = heappop(q)

            for n in self.roadmapedges[0][near[0]]:
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

    def visualize(self, solutiontrace):
        robot = self.env.GetRobots()[0]
        tracerobots = []
        
        for point in solutiontrace:
            robot.SetActiveDOFValues(point)
            time.sleep(0.1)
        
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
            self.color = (1,0,1)
            self.traceheight = 0.3
            self.roadmap.append([s])
            self.roadmapedges.append({0:[]})
            self.roadmap.append([g])
            self.roadmapedges.append({0:[]})
            self.buildtime = MAXTIME
            while (time.time()-self.starttime) <= MAXTIME:
                rand = self.sample_uniform()
                self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], status, new = self.extend(self.roadmap[self.currentgraph], self.roadmapedges[self.currentgraph], rand)
                if status != 'trapped':
                    connected = self.connectN(new)

                    if connected:
                        self.currentstate = 'connected graphs' 
                        break

                if self.currentstate == 'build graphs':
                    self.swapN()

        if self.currentstate == 'connected graphs':
            i_s = distance.cdist([s], self.roadmap[0]).argmin()
            i_g = distance.cdist([g], self.roadmap[0]).argmin()
            self.start = (i_s,s)
            self.goal = (i_g,g)

            path = self.search()
            if len(path) > 0:
                print "Path length: ", len(path)
                print "Roadmap size: ", len(self.roadmap[0])
                print "Completed in " + str(time.time()-self.starttime)
                if visualize:
                    path = path[::-1]
                    # for point in path:
                        # self.trace.append(self.env.plot3(points=[point[0], point[1], 0.3], pointsize=0.05, colors=array((1,1,1)), drawstyle=1))
                    time.sleep(0.1)
                    self.visualize(path)
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
            print 'out of time' + ',' + str(time.time()-self.starttime) + ',' + str(numstates)
            del self.trace
            self.trace = []
            return False, []

def setupRobot(robot):
    if robot.GetName() == 'robot1':
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

def main():

    global mode

    env = Environment()
    env.Load('envs/3d/'+envnum+'.dae')
    initial_transform = pickle.load(open(os.path.join('envs','3d','fetch_transforms', str(envnum) + '.pkl'), 'rb'))
    fetch_robot = fetch.FetchRobot(env, initial_transform)
    
    if visualize:
        env.SetViewer('qtcoin')
    
    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    collisionChecker.SetCollisionOptions(CollisionOptions.Contacts)
    env.SetCollisionChecker(collisionChecker)
    
    # load robots
    robot = env.GetRobots()[0]
    setupRobot(robot)
    env.UpdatePublishedBodies()

    jointlimits = robot.GetActiveDOFLimits()

    # tuck
    start = [ 0.00000000e+00,  1.32000492e+00,  1.39998343e+00, -1.99845334e-01,
    1.71996798e+00,  2.70622649e-06,  1.66000647e+00, -1.34775426e-06]

    # 7.0.1
    goals = { '7.0.1':  [ 0.15317293, -0.02241246,  0.3       ,  2.8       , -0.17657254, 1.89736047, -1.64968682,  1.10353645],
              '8.0.1':  [ 0.2194822 ,  0.68207005,  0.4       , -0.8       ,  1.86541826, 0.71582408, -1.54721034,  1.04590355]
    }
    goal = goals[test_sample]
    
    robot.SetActiveDOFValues(goal)
    robot.SetActiveDOFValues(start)
    print("start: ", start)
    print("goal: ", goal)

    pd = load_pd()
    if mode == 'llp':
        n = 250 # taken from average of number of samples generated in previous experiments
        m = 0
        total_success = 0
        for i in xrange(NUMRUNS):
            problem = LL(env,jointlimits,pd,n,m,start,goal)
            starttime = time.time()
            success, path = problem.plan(start,goal, visualize)
            if success:
                total_success += 1
            print success, total_success, i+1
            print 'LLP', MAXTIME
            print '*********'
        print 'Completed.'
        print test_sample, 'LLP', str(MAXTIME), total_success
        results_file.write(str(MAXTIME) + ',' + str(total_success) + '\n')
    else:
        n = 250
        m = 250
        total_success = 0
        for i in xrange(NUMRUNS):
            problem = LL(env,jointlimits,pd,n,m)
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
