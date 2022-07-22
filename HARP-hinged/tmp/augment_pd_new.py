import copy
import os2emxpath
import random
import scipy.spatial

from scipy.ndimage.measurements import label
import numpy as np
from data_structures import HLGraph, State
import tqdm
import cv2
import collections
import heapq


class PDAugmenter(object):
    def __init__(self,original_pd, mask_threshold, env,  connection_threshold = 0.5):
        '''
        params: 
        original_pd: original predictions from the network. just for the first layer. 
        threshold: threshold to be used to classify critical regions
        '''
        self.original_pd = original_pd
        self.count = 0
        self.path = None
        self.path_pd = None
        self.colors = None
        self.temp = None
        self.env = env
        self.flag = False
        self.centroids = {}
        self.mask_threshold = mask_threshold
        self.connection_threshold = (self.original_pd.shape[0] * connection_threshold) / 2.5
        self.mask = self.original_pd > self.mask_threshold
        self.updated_pd = self.original_pd.copy()
        self.updated_pd[self.mask] = 1.0
        self.updated_pd[~self.mask] = 0.0
        self.pd = self.updated_pd * self.env[:,:,0]
        hl_labels, num = label(self.updated_pd)
        self.num = num
        self.hl_labels = hl_labels
        self.process_hl_labels()
        self.kd_tree,self.node_kd_tree = self.__create_kd_tree()
        # self.centroids = {}
        self.hl_graph = self.__initialize_graph()
        # self.voronoi_map = np.zeros(shape = self.updated_pd.shape)
        # self.__create_voronoi()
        # self.voronoi_map = self.voronoi_map * self.env[:,:,0]
        # self.__update_voronoi()
        # assert self.voronoi_map.shape[0] == self.env.shape[0] and self.voronoi_map.shape[1] == self.env.shape[1]
        # self.hl_graph = self.__make_graph()

    
    def __initialize_graph(self):
        g = HLGraph()
        for i in range(1,self.num+1):
            idx = np.argwhere(self.hl_labels == i)
            if idx.shape[0] > 0:
                g.add_state(State(i,self.centroids[i]))
        return g

    def generate_sample(self):
        while True:
            i = np.random.choice(range(1,self.num+1))
            idx = np.argwhere(self.hl_labels == i)
            if idx.shape[0] > 0:
                id_list = idx.tolist()
                sample = random.choice(id_list)
                return sample

    def get_hl_state(self,cstate):
        x = cstate[0]
        y = cstate[1]
        dist, index = self.kd_tree.query([y,x])
        idx, idy = self.kd_tree.data[index]
        s = self.hl_labels[idx,idy]
        return s

    def check_path(self,start,goal):
        s1 = self.hl_graph.get_state(self.get_hl_state(start))
        s2 = self.hl_graph.get_state(self.get_hl_state(goal))
        try:
            path = self.hl_graph.get_path(s1,s2)
            flag = True
        except Exception, e:
            path = None
            solutions = self.beam_search(s1,s2)
            if len(solutions) == 0:
                return None, False
            path = set()
            for s in solutions:
                for state in s:
                    path.add(state.id)
            path = list(path)
            flag = False
        return path, flag

    def update_pd(self,path):
        pd = self.updated_pd.copy()
        mask = np.ma.masked_where(self.hl_labels == -1, self.pd).mask
        for i in path:
            m2 = np.ma.masked_where(self.hl_labels == i, self.pd).mask
            mask = np.ma.mask_or(mask,m2)
        
        pd[~mask] = 0.0
        pd[mask] = 1.0
        pd = pd / np.sum(pd)
        pd = pd / np.max(pd)
        return pd

    def add_exp(self,traj,start,goal):
        self.flag = True
        current = self.get_hl_state(start)
        for point in traj:
            # point[0], point[1] = point[1], point[0]
            closest = self.get_hl_state(point)
            if closest != current:
                self.add_edge(current,closest)
                current = closest
        
    def add_edge(self,s1,s2):
        n1 = self.hl_graph.get_state(s1)
        n2 = self.hl_graph.get_state(s2)
        if not self.hl_graph.has_edge(s1,s2):
            self.hl_graph.add_connection(n1,n2)

    def is_neighbor(self,c1,c2):
        s1 = self.get_hl_state(c1)
        s2 = self.get_hl_state(c2)
        return self.hl_graph.has_edge(s1,s2)
    
    def is_in_same_state(self,c1,c2):
        return self.get_hl_state(c1) == self.get_hl_state(c2)

    def __create_kd_tree(self):
        idx = np.argwhere(self.hl_labels)
        # # for i in range(idx.shape[0]):
        # #     if self.hl_labels[idx[i,0],idx[i,1]] == 0:
        # #         print "here"
        # l = []
        # for i in range(1,self.num+1):
        #     idx = np.argwhere(self.hl_labels == i)
        #     if idx.shape[0] > 0:
        #         mean = np.median(idx,axis=0).astype(np.int)
        #         # l.append(mean)
        #         # while self.hl_labels[mean[0],mean[1]] == 0:
        #         #     mean[0] += 1
        #         #     mean[1] += 1
        #         l.append(mean)
        #         self.centroids[i] = mean
        kd_tree = scipy.spatial.KDTree(idx)
        for i in range(1,self.num+1):
            idx = np.argwhere(self.hl_labels == i)
            if idx.shape[0] > 0:
                mean = np.mean(idx,axis=0).astype(np.int)
                self.centroids[i] = mean
        node_kd_tree = scipy.spatial.KDTree(self.centroids.values())
        return kd_tree,node_kd_tree


    def plot(self,fname = None, mp = []):
        # if path is None:
        #     path = range(1,self.num+1)
        #     flag = False
        if self.colors is None:
            self.colors = {0:[0,0,0]}
            for i in range(1,self.num+1):
                self.colors[i] = [np.random.random(),np.random.random(),np.random.random()]
        img = np.ones(shape=(self.hl_labels.shape[0],self.hl_labels.shape[1],3))
        img = img * self.env[:,:,:3]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if self.hl_labels[i,j] != 0:
                    img[i,j,:] = self.colors[self.hl_labels[i,j]]
        for state in self.hl_graph.states:
            x,y = state.get_centroid()
            # x,y = state.get_centroid()
            cv2.circle(img, (y, x), 10, [0, 0, 0])
            cv2.putText(img,str(state.id), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1,0,1), 2, cv2.LINE_AA)

        drawn = set()
        for s1 in self.hl_graph.states:
            for s2 in self.hl_graph.states:
                if self.hl_graph.has_edge(s1,s2) and (s1,s2) not in drawn:
                    # y1,x1 = s1.get_centroid()
                    x1,y1 = s1.get_centroid()
                    # y2,x2 = s2.get_centroid()
                    x2,y2 = s2.get_centroid()
                    cv2.line(img,(y1,x1), (y2,x2), [0, 0, 0, ])
        for wp in mp:
            img[wp[1],wp[0],:] = [1,0,0]

        img = img * 255.0
        if fname is None:
            fname = "temp"
        cv2.imwrite(fname+".png",img)


    def euclidean_distance(self,p1,p2):
        return ((p1[0]-p2[0])**2.0 + (p1[1] - p2[1])**2 ) ** 0.5

    def estimate_cost(self,s1,s2):
        if s1 == s2:
            return 0
        elif self.hl_graph.has_edge(s1,s2):
            return 0
        else:
            p1 = s1.get_centroid()
            p2 = s2.get_centroid()
            return self.euclidean_distance(p1,p2)

    def estimate_hueristic(self,s1,end):
        if s1 == end:
            return 0
        try:
            path = self.hl_graph.get_path(s1,end)
        except Exception,e:
            p1 = s1.get_centroid()
            p2 = end.get_centroid()
            return self.euclidean_distance(p1,p2)
        else:
            return 0


    def beam_search(self,start, end):
        '''
        start: a high-level state (ds: Node)
        end: a high-level state (ds:Node)
        '''
        k = 20
        fringe = []
        path = []
        visited = set()
        heapq.heappush(fringe,(0,[start,path,visited]))
        solutions = []
        # while k > 0 and len(fringe) > 0: # add one more terminating condition
        while len(fringe) > 0: # add one more terminating condition
            new_fringe = []
            i = 0
            while i < 10 and i < len(fringe):
                new_fringe.append(heapq.heappop(fringe))
                i+= 1
            fringe = []
            while len(new_fringe) > 0:
                current,path,visited = heapq.heappop(new_fringe)[1]
                if current not in visited:
                    visited.add(current)
                    path.append(current)
                    if current.id != end.id:
                        dist, index = self.node_kd_tree.query([current.get_centroid()],k=len(self.node_kd_tree.data))
                        p = 0
                        ii = 0
                        while ii < len(index[0]):
                            idx, idy = self.node_kd_tree.data[index[0][ii]]
                            # s = self.hl_graph.get_state(self.hl_labels[idx,idy])
                            s = self.get_hl_state([idy,idx])
                            s = self.hl_graph.get_state(s)
                            if s not in visited:
                                # print "Adding node {} to fring for parent {}".format(s.id, current.id)
                                total_cost = self.estimate_cost(s,current) + self.estimate_hueristic(s,end)
                                heapq.heappush(fringe,(total_cost,[s,path[:],copy.deepcopy(visited)]))
                                if total_cost == 0:
                                    break
                                p += 1
                            else:
                                # print "Not adding node {} to fring for parent {} as it is already visited".format(s.id, current.id)
                                pass
                            ii += 1
                            # print "Fringe size: {}".format(len(fringe))

                    else:
                        solutions.append(path)
                        ids = [l.id for l in path]
                        # print "Path found: {}".format(ids)
                        k -= 1
        return solutions




    # def get_cost_function(self,hl_graph,goal):
    #     def s(s1):
    #         local_goal = goal.get_centroid()
    #         s1 = s1.hl_graph.get_state(s1).get_centroid()
    #         return ((s1[0]-local_goal[0])**2 + (s1[1] - local_goal[1]) ** 2 )**0.5
    #     return s








    # def generate_sample(self,pd = False):
    #     n = np.random.choice(self.path)
    #     if pd:
    #         mask = np.ma.masked_where(self.hl_labels == n, self.hl_labels)
    #         pd = self.hl_labels.copy()
    #         pd[~mask.mask] = 0.0
    #         pd[mask.mask] = 1.0
    #     else:
    #         # mask = np.ma.masked_where(self.voronoi_map == -1, self.voronoi_map).mask
    #         # for s in self.path:
    #         #     m2 = np.ma.masked_where(self.voronoi_map == s, self.voronoi_map).mask
    #         #     mask = np.ma.mask_or(mask,m2)
    #         mask = np.ma.masked_where(self.voronoi_map == n, self.voronoi_map).mask
    #         pd = self.voronoi_map.copy()
    #         pd[~mask] = 0.0
    #         pd[mask] = 1.0

    #     # pd = self.temp.copy()

    #     xy_pd = pd.reshape((pd.shape[0] * pd.shape[1],))
    #     xy_pd = xy_pd / float(np.sum(xy_pd))
    #     xy_sample = np.random.choice(range(xy_pd.shape[0]),p = xy_pd)
    #     y_dof = int(xy_sample/pd.shape[1])
    #     x_dof = xy_sample - int(y_dof * pd.shape[1])

    #     return x_dof, y_dof



    # def __update_voronoi(self):
    #     print "converting voronois to SCCs."
    #     new_num = self.num
    #     new_voronoi = copy.deepcopy(self.voronoi_map)
    #     for i in tqdm.tqdm(range(1,self.num+1)):
    #         vornonoi_map_copy = copy.deepcopy(self.voronoi_map)
    #         mask = np.ma.masked_where(vornonoi_map_copy == i,vornonoi_map_copy)
    #         vornonoi_map_copy[~mask.mask] = 0.0
    #         vornonoi_map_copy[mask.mask] = 1.0
    #         hl_labels, num = label(vornonoi_map_copy)
    #         if num > 1:
    #             mask = np.ma.masked_where(self.hl_labels == i, self.hl_labels)
    #             self.hl_labels[mask.mask] = 0
    #             mask = np.ma.masked_where(hl_labels == 1, self.update_pd)
    #             idx = np.argwhere(mask.mask)
    #             sidx = np.random.choice(range(idx.shape[0]),int(idx.shape[0] * 0.15))
    #             sids = idx[sidx]
    #             for x,y in sids:
    #                 self.updated_pd[x,y] = 1.0
    #                 self.hl_labels[x,y] = i
    #             for j in range(2,num+1):
    #                 mask = np.ma.masked_where(hl_labels == j, new_voronoi)
    #                 new_voronoi[mask.mask] = new_num + 1
    #                 new_num += 1
    #                 mask = np.ma.masked_where(hl_labels == j, self.updated_pd)
    #                 idx = np.argwhere(mask.mask)
    #                 sidx =  np.random.choice(range(idx.shape[0]),int(idx.shape[0] * 0.15))
    #                 sids = idx[sidx]
    #                 for x,y in sids:
    #                     self.updated_pd[x,y] = 1.0
    #                     self.hl_labels[x,y] = new_num

    #     self.num = new_num
    #     self.voronoi_map = copy.deepcopy(new_voronoi)



    
    def process_hl_labels(self):
        count = collections.Counter(list(self.hl_labels.flatten()))
        for n in range(self.num+1):
            if count[n] < 35:  # TODO: some better approach to do this.
                idx = np.where(self.hl_labels == n)
                self.hl_labels[idx] = 0


    # def __create_voronoi(self):
    #     print "generating voronoi..."
    #     pbar = tqdm.tqdm(total = self.updated_pd.shape[0]*self.updated_pd.shape[1])
    #     for i in range(self.updated_pd.shape[0]):
    #         for j in range(self.updated_pd.shape[1]):
    #             s = self.get_closest_state([i,j])
    #             # if s == 0:
    #             #     print "here.."
    #             self.voronoi_map[i,j] = s
    #             # self.voronoi_map[i,j] = self.get_closest_state([i,j])
    #             pbar.update(1)
    #     print "inconsistent: {}".format(self.count)

    # def search_high_level_plan(self):
    #     pass


    # def plot(self,path = None,graph = True):
    #     flag = True
    #     if path is None:
    #         path = range(1,self.num+1)
    #         flag = False
    #     if self.colors is None:
    #         self.colors = {0:[0,0,0]} 
    #         for i in range(1,self.num+1):
    #             self.colors[i] = [np.random.random(),np.random.random(),np.random.random()]
    #     img = np.zeros(shape=(self.hl_labels.shape[0],self.hl_labels.shape[1],3))
    #     vimg = copy.deepcopy(img)
    #     for i in range(img.shape[0]):
    #         for j in range(img.shape[1]):
    #             if self.hl_labels[i,j] in path:
    #                 img[i,j,:] = self.colors[self.hl_labels[i,j]]
    #             if self.voronoi_map[i,j] in path:
    #                 vimg[i,j,:] = self.colors[int(self.voronoi_map[i,j])]

    #     # vimg = vimg * self.env[:,:,:3]
    #     if graph:
    #         vimg = self.draw_connections(vimg)

    #     if flag:
    #         cv2.imwrite("temp_path.png",img*255.0)
    #         cv2.imwrite("vtemp_path.png",vimg*255.0)
    #     else:
    #         cv2.imwrite("temp.png", img * 255.0)
    #         cv2.imwrite("vtemp.png", vimg * 255.0)

    # def draw_connections(self, img):
    #     labels = list(set(self.voronoi_map.flatten().tolist()))
    #     means = {}
    #     for i in labels:
    #         idx = np.argwhere(self.voronoi_map == i)
    #         mean = np.mean(idx,axis = 0)
    #         x,y = int(mean[0]),int(mean[1])
    #         means[i] = (y,x)
    #         cv2.circle(img,(y,x),5,[0,0,0])
    #     drawn = set()
    #     for i in labels:
    #         for j in labels: 
    #             if self.hl_graph.has_edge(iget_state,j):
    #                 if (i,j) not in drawn:
    #                     cv2.line(img,means[i],means[j],[0,0,0,])
    #                     drawn.add((i,j))
    #                     drawn.add((j,i))
    #     return img
                
        

    # def __make_graph(self):
    #     g = HLGraph()
    #     for i in range(1,self.num+1):
    #         g.add_state(State(i))
    #     pbar = tqdm.tqdm(total= self.updated_pd.shape[0]-1 * self.updated_pd.shape[1]-1)
    #     updated_voronoi = self.voronoi_map * self.env[:,:,0]
    #     self.voronoi_map = updated_voronoi
    #     # for i in range(1,self.num+1):
    #     #     for j in range(1,self.num+1):
    #     #         if j != i:
    #     #             if self.check_connectivity(i,j):
    #     #                 g.add_connection(g.get_state(i),g.get_state(j))
    #     #         pbar.update(1)
    #     visited = set()
    #     for i in range(self.updated_pd.shape[0]-1):
    #         for j in range(self.updated_pd.shape[1]-1):
    #             if self.voronoi_map[i,j] != self.voronoi_map[i,j+1] and self.voronoi_map[i,j] != 0 and self.voronoi_map[i,j+1] != 0:
    #                 if (self.voronoi_map[i,j],self.voronoi_map[i,j+1]) not in visited:
    #                     g.add_connection(g.get_state(self.voronoi_map[i,j]),g.get_state(self.voronoi_map[i,j+1]))
    #                     visited.add((self.voronoi_map[i,j],self.voronoi_map[i,j+1]))
    #                     visited.add((self.voronoi_map[i,j+1],self.voronoi_map[i,j]))
    #             elif self.voronoi_map[i,j] != self.voronoi_map[i+1,j] and self.voronoi_map[i,j] != 0 and self.voronoi_map[i+1,j] != 0:
    #                 if (self.voronoi_map[i,j],self.voronoi_map[i+1,j]) not in visited:
    #                     g.add_connection(g.get_state(self.voronoi_map[i,j]),g.get_state(self.voronoi_map[i+1,j]))
    #                     visited.add((self.voronoi_map[i+1,j],self.voronoi_map[i,j]))
    #                     visited.add((self.voronoi_map[i,j],self.voronoi_map[i+1,j]))
    #             pbar.update(1)

    #     return g

    # def check_connectivity(self,i,j):
    #     idi = np.argwhere(self.hl_labels == i)
    #     idj = np.argwhere(self.hl_labels == j)
    #     for point_i in idi: 
    #         for point_j in idj: 
    #             if (((point_i - point_j)**2).sum())**(0.5) < self.connection_threshold:
    #                 return True
    #     return False

    # def generate_distribution(self,start,goal):
    #     s1 = self.hl_graph.get_state(self.check_voronoi(start))
    #     s2 = self.hl_graph.get_state(self.check_voronoi(goal))
    #     self.path = self.hl_graph.get_path(s1,s2)
    #     pd = self.updated_pd.copy()
    #     for i in range(pd.shape[0]):
    #         for j in range(pd.shape[1]):
    #             if self.hl_labels[i,j] not in self.path:
    #                 pd[i,j] = 0
    #     pd = pd / np.sum(pd)
    #     pd = pd / np.max(pd)
    #     self.path_pd = pd

    #     mask = np.ma.masked_where(self.voronoi_map == -1, self.voronoi_map).mask
    #     for s in self.path:
    #         m2 = np.ma.masked_where(self.voronoi_map == s, self.voronoi_map).mask
    #         mask = np.ma.mask_or(mask, m2)
    #     temp = self.voronoi_map.copy()
    #     temp[~mask] = 0.0
    #     temp[mask] = 1.0

    #     self.temp = temp


    #     return pd, self.path
    
    # def nearest_nonzero_idx(self,a,x,y):
    #     # idx_o = np.argwhere(a)
    #     # idx_1 = idx_o[((idx_o - [x,y])**2).sum(1).argmin()]
    #     # temp = self.kd_tree.query([[x,y]])
    #     dist, index = self.kd_tree.query([[x,y]])
    #     idx, idy = self.kd_tree.data[index[0],:]
    #     # if idx != idx_1[0] or idy != idx_1[1]:
    #     #     self.count += 1
    #     return int(idx), int(idy)

    # def get_closest_state(self,cstate):
    #     xbin, ybin = cstate[0], cstate[1]
    #     idx, idy = self.nearest_nonzero_idx(self.hl_labels,xbin,ybin)
    #     return self.hl_labels[idx,idy]

    # def check_voronoi(self,cstate):
    #     xbin, ybin = cstate[1], cstate[0]
    #     return self.voronoi_map[xbin,ybin]

    # def get_path(self,start,goal):
    #     s1 = self.hl_graph.get_state(self.check_voronoi(start))
    #     s2 = self.hl_graph.get_state(self.check_voronoi(goal))
    #     path = self.hl_graph.get_path(s1,s2)
    #     return path


    
    
if __name__ == "__main__":
    pd = np.squeeze(np.load("./network/results/8.0.1.npy"))
    # pd = np.squeeze(np.load("./network/results/10.2.1.npy"))
    # pd = np.squeeze(np.load("./network/results/10.2.1.npy"))
    xy_pd = pd[:,:,0]
    env = np.squeeze(np.load("./test/env8.0/8.0.1.npy"))
    # env = np.squeeze(np.load("./test/env10.2/10.2.1.npy"))
    # env = np.squeeze(np.load("./test/env10.2/10.2.1.npy"))
    pda = PDAugmenter(xy_pd,0.6,env,0.2)
    # start = [-1.6, -1.0, 0]
    # end = [2.0, 1.0, 1.57]
    # path = pda.get_path(start,end)
    # pda.plot()
    pass
    
