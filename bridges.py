# Python program to find bridges in a given undirected graph
#Complexity : O(V+E)
  
from collections import defaultdict
import numpy as np
import sys
from time import time  

#This class represents an undirected graph using adjacency list representation
class Graph:
  
    def __init__(self, vertices = 0):
        self.V = vertices
        self.graph = defaultdict(list) # default dictionary to store graph
        self.Time = 0
        self.Bridges = list()
  

    def count_edges(self):
        return sum(len(v[1]) for v in self.graph.items())/2

    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def import_net(self, file_name):
        print(file_name)
        mt = np.genfromtxt('networks/' + file_name + '.csv',delimiter=',').astype(np.int32)
        for row in mt:
            source = row[0]
            target = row[1]
            self.addEdge(source, target)
        all_nodes = set(mt.flatten())
        self.V = len(all_nodes)
        print ("Adicionado todas as arestas ao grafo")
  
    def bridgeUtil(self,u, visited, parent, low, disc):
        #Count of children in current node 
        children =0
 
        # Mark the current node as visited and print it
        visited[u]= True
 
        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
 
        #Recur for all the vertices adjacent to this vertex
        for v in self.graph[u]:
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if visited[v] == False :
                parent[v] = u
                children += 1
                self.bridgeUtil(v, visited, parent, low, disc)
 
                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    # print ("%d %d" %(u,v))
                    self.Bridges.append((u,v))
     
                     
            elif v != parent[u]: # Update low value of u for parent function calls.
                low[u] = min(low[u], disc[v])
 
 
    # DFS based function to find all bridges. It uses recursive
    # function bridgeUtil()
    def bridge(self):
  
        # Mark all the vertices as not visited and Initialize parent and visited, 
        # and ap(articulation point) arrays
        visited = [False] * (self.V)
        disc = [float("Inf")] * (self.V)
        low = [float("Inf")] * (self.V)
        parent = [-1] * (self.V)
 
        # Call the recursive helper function to find bridges
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if visited[i] == False:
                self.bridgeUtil(i, visited, parent, low, disc)

if __name__ == "__main__":
    g = Graph(6)
    g.addEdge(0,1)
    g.addEdge(1,2)
    g.addEdge(1,5)
    g.addEdge(2,3)
    g.addEdge(3,4)
    g.addEdge(3,5)
    t0 = time()
    g.bridge()
    print(g.Bridges)
    print("Running time: {}".format(time()-t0))
    # #  print(g.count_edges()/2)
    # g = Graph()
    # g.import_net("real2")
    # num_nodes = g.V
    # print("Nodes in G: " + str(num_nodes))
    # num_edges = g.count_edges()
    # print("Edges in G: " + str(num_edges))
    # sys.setrecursionlimit((num_nodes + num_edges)*2)
    # g.bridge()
    # print (g.bridges[:10])