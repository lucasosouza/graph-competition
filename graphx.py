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

        # define a list to keep nodes ordering
        self.order = 1
        self.nodes_ordering = [0] * self.V

        assert (self.V == len(self.graph.keys())), "Número de nós correto"
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

        # update a table telling in which cluster each node is
        self.nodes_cluster = [0] * (self.V)

        # reset clusters
        self.clusters = []
        not_visited = np.ones(self.V)        
        # loop through all vertices
        idx_cluster = 0
        for start_node in range(self.V):
            # if still not visited
            if not_visited[start_node] == 1:
                # set not_visited to False
                not_visited[start_node] = 0
                # init connected component list of nodes
                cc = [start_node]
                # init queue
                queue = [start_node]
                # update position
                self.nodes_cluster[start_node] = idx_cluster
                # iterate
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
                            # update position
                            self.nodes_cluster[node2] = idx_cluster
                # print("Length of cluster: {}".format(len(cc))) 
                idx_cluster += 1
                self.clusters.append(cc)

    def pre_clusters_size(self):
        """ Populate array with cluster sizes
        Speed up bridge score implementation """

        num_clusters = len(self.clusters)

        # populate array with cluster sizes
        self.cluster_size = [0] * (num_clusters)
        for idx, cluster in enumerate(self.clusters):
            self.cluster_size[idx] = len(cluster)

        self.cluster_bridges = []
        for _ in range(num_clusters):
            self.cluster_bridges.append([])

        # find the cluster for each node
        for node1, node2 in self.bridges:

            node1_cluster = self.nodes_cluster[node1]
            node2_cluster = self.nodes_cluster[node2]

            # appends - origin node and which cluster it conects
            self.cluster_bridges[node1_cluster].append((node1, node2_cluster))
            self.cluster_bridges[node2_cluster].append((node2, node1_cluster))


    def score_bridges(self, base_score=0.5):
        """  Calculate score for bridges"""

        # divided by 1e4 took 40 seconds
        # divided by 1e3 should take 400 seconds, or about 7 minutes
        for i in range(int(len(self.bridges)/1e3)):

            # get largest cluster
            largest_cluster = np.argmax(self.cluster_size)

            # find the most significant bridge
            max_size = 0
            most_significant_bridge = None

            # early stopping, when largest node has no more bridges
            if len(self.cluster_bridges[largest_cluster]) == 0:
                break

            for idx, bridge in enumerate(self.cluster_bridges[largest_cluster]):
                origin_node, target_cluster = bridge
                target_cluster_size = self.cluster_size[target_cluster]
                if target_cluster_size > max_size:
                    max_size = target_cluster_size
                    most_significant_bridge = idx

            # remove bridge from cluster list
            selected_bridge = self.cluster_bridges[largest_cluster].pop(most_significant_bridge)
            selected_node, selected_target_cluster = selected_bridge

            # add the origin node to list
            if self.nodes_ordering[selected_node] == 0:
                self.nodes_ordering[selected_node] = self.order
                self.order += 1

            # update size of the bridge
            new_size = max(0, self.cluster_size[largest_cluster] - (1+self.cluster_size[selected_target_cluster]))
            self.cluster_size[largest_cluster] = new_size

            # iterate

    def score_remaining(self, base_score=0.5):
        """  Calculate score for remaining nodes """

        node_cluster_size = [0] * self.V

        for idx, cluster in enumerate(self.clusters):
            len_cluster = self.cluster_size[idx]
            for node in cluster:
                node_cluster_size[node] = len_cluster
                # check if node is not already classified
                if self.nodes_ordering[node] == 0:
                    len_cluster -= 1

        connections = [len(k[1]) for k in self.graph.items()]
        final = list(zip(list(range(self.V)), connections, node_cluster_size))
        remaining_nodes = sorted(final, key=lambda x:(-x[2], -x[1]))
        # remaining_nodes = sorted(final, key=lambda x:(-x[1], -x[2]))
        for node, connections, cluster_size in remaining_nodes:
            if self.nodes_ordering[node] == 0:
                self.nodes_ordering[node] = self.order
                self.order += 1

    def export(self):

        to_export = zip(range(self.V), self.nodes_ordering)
        to_export = sorted(list(to_export), key=lambda x:x[1])

        return to_export


   # def score_remaining(self, base_score=0.5):
   #      """  Calculate score for remaining nodes """

   #      # while there are unordered clusters
   #      while min(self.nodes_ordering) == 0:

   #          # get largest clusters            
   #          largest_cluster = np.argmax(self.cluster_size)
   #          second_largest_cluster = \
   #              np.argmax(self.cluster_size[0:largest_cluster] + self.cluster_size[largest_cluster+1:])
   #          if second_largest_cluster >= largest_cluster:
   #              second_largest_cluster += 1

   #          diff = self.cluster_size[largest_cluster] - self.cluster_size[second_largest_cluster]

   #          # early stopping
   #          if diff < 1e6:
   #              break

   #          nodes_connections = \
   #              [(node, len(self.graph[node])) for node in self.clusters[largest_cluster]]
   #          nodes_connections = sorted(nodes_connections, key=lambda x:-x[1])

   #          for i in range(diff):
   #              # append node to ordered list
   #              if self.nodes_ordering[selected_node] == 0:
   #                  self.nodes_ordering[nodes_connections[i][0]] = self.order
   #                  self.order += 1

   #              # reduce size of cluster
   #              self.cluster_size[largest_cluster] -= 1
