# Python program to find bridges in a given undirected graph
#Complexity : O(V+E)
  
from collections import defaultdict
import numpy as np
import sys
from time import time
  
#This class represents an undirected graph using adjacency list representation
class Graph:
  
    def __init__(self, vertices = 0):
        self.V = vertices # integer?
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
            node1 = row[0]
            node2 = row[1]
            self.addEdge(node1, node2)
        all_nodes = set(mt.flatten())
        self.V = len(all_nodes)
        print ("Adicionado todas as arestas ao grafo")
  
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
                    self.Bridges.append((u,v))
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
    def bridge(self):
  
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

if __name__ == "__main__":
    # g = Graph(6)
    # g.addEdge(0,1)
    # g.addEdge(1,2)
    # g.addEdge(2,3)
    # g.addEdge(3,4)
    # g.addEdge(3,5)
    # g.addEdge(1,5)
    # t0 = time()
    # g.bridge()
    # print(g.Bridges)
    # print("Running time: {}".format(time()-t0))
    g = Graph()
    g.import_net("real2")
    print("Nodes in G: " + str(g.V))
    g.bridge()
    print ("Total Edges: " + str(g.count_edges()))
    print ("Total Bridges: " + str(len(g.Bridges)))