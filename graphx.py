import numpy as np
from time import time
from utils import import_net

from collections import defaultdict
from time import time
  
class Graph:  
    """ Represents an undirected graph using adjacency list representation """

    def __init__(self, vertices = 0, filename=None):
        self.V = vertices # integer?
        self.graph = defaultdict(list) # default dictionary to store graph
        self.Time = 0
        self.bridges = set()
        self.edges = set()
        self.sparse_graph = defaultdict(list)
        self.clusters=[]
        if filename:
            self.import_net(filename)

    def import_net(self, filename):
        print(filename)
        max_node = 0
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = [int(i) for i in line.split(",")]
                node1 = line[0]
                node2 = line[1]
                self.addEdge(node1, node2)
                max_node = max(max_node, node1, node2)

        self.V = max_node+1
        print ("Adicionado todas as arestas ao grafo")

    def create_sparse_graph(self):
        sparse_edges = self.edges.difference(self.bridges)
        for node1, node2 in sparse_edges:
            # edges already duplicate when loaded
            self.sparse_graph[node1].append(node2)

    def countEdges(self):
        return len(self.edges)

    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.edges.add((u, v))
        self.edges.add((v, u))

    def bridgeUtil(self,u, visited, parent, children, low, disc):

        # create a stack
        stack = [u]
        backtrack_v = None

        while len(stack) > 0:

            # get top element of stack, without removing from stack
            u = stack[-1]

            done = True
     
            # only update time if not visited
            if not visited[u]:

                # Mark the current node as visited and print it
                visited[u]= True
         
                # Initialize discovery time and low value
                disc[u] = self.Time
                low[u] = self.Time
                self.Time += 1

            if backtrack_v:
                v = backtrack_v
                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    # print ("%d %d" %(u,v))
                    # need to be duplicated
                    self.bridges.add((u,v))
                    self.bridges.add((v,u))
                backtrack_v = None

            # else:
            for v in self.graph[u][children[u]:]:
                # If v is not visited yet, then make it a child of u
                # in DFS tree and recur for it
                if visited[v] == False :
                    parent[v] = u
                    children[u] += 1
                    stack.append(v)
                    # after appending, break from loop
                    done = False
                    break

                elif v != parent[u]: # Update low value of u for parent function calls.
                    low[u] = min(low[u], disc[v])

            if done:
                backtrack_v = stack.pop()
      
    # DFS based function to find all bridges. It uses recursive
    # function bridgeUtil()
    def find_bridges(self):
        """ Find bridges in a given undirected graph. Complexity : O(V+E) """
  
        # Mark all the vertices as not visited and Initialize parent and visited, 
        # and ap(articulation point) arrays
        visited = [False] * (self.V)
        disc = [float("Inf")] * (self.V)
        low = [float("Inf")] * (self.V)
        parent = [-1] * (self.V)
        children = [0] * (self.V)
 
        # Call the recursive helper function to find bridges
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if visited[i] == False:
                # print("Call bridgeUtil on {}".format(i))
                self.bridgeUtil(i, visited, parent, children, low, disc)

    def find_clusters(self):
        """ Connected components without using sets """

        # reset clusters
        self.clusters = []
        not_visited = np.ones(self.V)        
        # loop through all vertices
        for start_node in range(self.V):
            # if still not visited
            if not_visited[start_node] == 1:
                # set not_visited to False
                not_visited[start_node] = 0
                # init connected component list of nodes
                cc = [start_node]
                # init queue
                queue = [start_node]
                while len(queue) > 0:
                    node1 = queue.pop(0)
                    for node2 in self.sparse_graph[node1]:
                        if not_visited[node2]:
                            # set not_visited to False
                            not_visited[node2] = 0
                            # append to connected component
                            cc.append(node2)
                            # append to queue
                            queue.append(node2)
                # print("Length of cluster: {}".format(len(cc))) 
                self.clusters.append(cc)

    def pre_clusters_size(self):
        """ Populate array with cluster sizes
        Speed up bridge score implementation """

        # populate array with cluster sizes
        self.cluster_size = list(range(self.V))
        for cluster in self.clusters:
            len_cluster = len(cluster)
            for node in cluster:
                self.cluster_size[node] = len_cluster

    def score_bridges(self):
        """  Calculate score for bridges"""

        scored_bridges = []
        bridge_nodes = set()
        for node1, node2 in self.bridges:
            size1 = self.cluster_size[node1]
            size2 = self.cluster_size[node2]
            # using harmonic average
            score = (size1*size2)/(size1+size2)
            if size1 >= size2:
                if node1 not in bridge_nodes:
                    scored_bridges.append((node1, score))
                    bridge_nodes.add(node1)
            else: 
                if node2 not in bridge_nodes:
                    scored_bridges.append((node2, score))
                    bridge_nodes.add(node2)

        # can optimize a tiny bit applying set to scored_bridges instead. 
        self.scored_bridges = sorted(list(scored_bridges), key=lambda x:-x[1])

    def score_remaining(self):
        """  Calculate score for remaining nodes """

        # print("scored_bridges: {}".format(len(self.scored_bridges)))
        bridge_nodes = set([x[0] for x in self.scored_bridges])
        # print("bridge_nodes: {}".format(len(bridge_nodes)))
        all_nodes = set(self.graph.keys())
        # print("all_nodes: {}".format(len(all_nodes)))
        remaining_nodes = all_nodes.difference(bridge_nodes)
        # print("remaining_nodes: {}".format(len(remaining_nodes)))

        partial_graph = {node: self.graph[node] for node in remaining_nodes}

        # calculate variables
        X = map(lambda x:[x[0], list(x[1])], list(partial_graph.items()))
        X = map(lambda x:[x[0], len(x[1]), np.std((x[1]))], X)
        X = np.array(list(X))

        # mean of stds
        max_std = np.max(X[:,2])
        min_std = np.min(X[:, 2])
        den = max_std - min_std

        # apply score
        X = map(lambda x:[x[0], x[1], x[2], (x[2]-min_std)/den], X)
        X = map(lambda x:(int(x[0]), x[1]*(0.5+x[3])), X)
        X = list(X)
        
        self.scored_remaining = sorted(X, key=lambda x:-x[1])
        # print(len(self.scored_remaining))

    def export(self):
        print(len(self.scored_bridges))
        print(len(self.scored_remaining))
        all_scored = self.scored_bridges + self.scored_remaining
        print(len(all_scored))
        return all_scored
