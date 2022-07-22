import networkx as nx

class State():
    def __init__(self,id,ll_state = None):
        self.id = id
        self.ll_state = ll_state

    def __hash__(self):
        return self.id

    def __eq__(self, o):
        return self.id == o.id

    def get_centroid(self):
        return self.ll_state

class HLGraph(object):
    def __init__(self):
        self.g = nx.Graph()
        self.states = []
        self.connections = {}
    
    def add_state(self,state):
        self.g.add_node(state.id)
        self.states.append(state)
    
    def add_connection(self,src, dest):
        if src not in self.connections:
            self.connections[src] = []
        self.connections[src].append(dest)
        self.g.add_edge(src.id,dest.id)

    def is_path(self,src,dest):
        return nx.has_path(self.g,src.id,dest.id)

    def get_path(self,src,dest): 
        return nx.shortest_path(self.g,src.id,dest.id)

    def get_state(self,id):
        for s in self.states:
            if s.id == id:
                return s
        return None
    def has_edge(self,s1,s2):
        if type(s1) == type(State(1)):
            s1 = s1.id
        if type(s2) == type(State(1)):
            s2 = s2.id
        return self.g.has_edge(s1,s2)



if __name__ == "__main__":
    s1 = State(1)
    s2 = State(2)
    s3 = State(3)
    G = HLGraph()
    G.add_state(s1)
    G.add_state(s2)
    G.add_state(s3)
    G.add_connection(s1,s2)
    print (G.is_path(s1,s2))
    print (G.is_path(s1,s3))
    print (G.is_path(s2,s1))

    ## use scipy.ndimage.measurements.label to identify regions in the predicted distribution. 
    ## use those regions to generate a high level state space and use shortest path in that state space to prune  the distribution. 
