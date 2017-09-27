import numpy as np
from time import time
from utils import import_net

from collections import defaultdict
from time import time

from heap import Heap
  
class Graph:  
    """ Represents an undirected graph using adjacency list representation """

    def __init__(self, vertices = 0, filename=None):
        self.graph = defaultdict(list) # default dictionary to store graph
        self.Time = 0
        self.V = vertices # integer?
        if filename:
            self.import_net(filename)
        self.ci = [-1] * self.V

    def compute_ci(self):
        for node in range(self.V):
            if self.ci[node] < 0:
                self.ci[node] = self.compute_ci_node(node, 2)

    def compute_ci_node(self, start_node, dist):

        visited = [0] * self.V
        visited[start_node] = 1 
        deg = 0
        counters = [0] * (dist+2)
        edges = [0] * (dist+2)
        edges[0] = len(self.graph[start_node])
        counters[0] = len(self.graph[start_node])
        queue = [start_node]
        # borders = []
        count_nodes = [0] * (dist+1)
        for i in range(dist+1):
            borders.append([])
        while len(queue) != 0:
            node = queue.pop(0)
            for adj_node in self.graph[node]:
                # print("adj_node: ", adj_node)
                counters[deg] -= 1
                # if not visited
                if visited[adj_node] == 0:
                    # mark as visited, and add to queue
                    count_nodes[deg] += 1
                    visited[adj_node] = 1
                    queue.append(adj_node)
                    # borders[deg].append(adj_node)
                    counters[deg+1] += len(self.graph[adj_node])
                    edges[deg+1] += len(self.graph[adj_node])
            if counters[deg] == 0:
                deg += 1
                if deg == (dist+1):
                    ci = (edges[0]-1)*(edges[deg] - count_nodes[deg-1])
                    # print("edges: ", edges)
                    # print("borders: ", borders)
                    # print("count_nodes: ", count_nodes)
                    return ci


    def create_max_heap(self):
        # heap index starts at 1
        self.heap = Heap([(0,0)] + list(zip(range(self.v), self.ci)))
        self.heap.build_max_heap()

    def remove_largest_value(self):
        node, max_value = self.heap.heap_extract_max()
        for adj_node in node:
            self.graph[node].remove(node)
        del self.graph[node]

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

        assert (self.V == len(self.graph.keys())), "Número de nós incorreto"
        print ("Adicionado todas as arestas ao grafo")

    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)


    def export(self):

        to_export = zip(range(self.V), self.nodes_ordering)
        to_export = sorted(list(to_export), key=lambda x:x[1])

        return to_export

g = Graph(9)
g.addEdge(0,1)
g.addEdge(0,2)
g.addEdge(1,2)
g.addEdge(1,5)
g.addEdge(2,3)
g.addEdge(2,4)
g.addEdge(2,6)
g.addEdge(2,5)
g.addEdge(5,6)
g.addEdge(5,8)
g.addEdge(6,8)
g.addEdge(6,7)
g.addEdge(6,4)
g.addEdge(6,3)
g.addEdge(3,7)
g.addEdge(3,4)
g.addEdge(4,7)

print(g.compute_ci())
print(g.ci)
