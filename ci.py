import numpy as np
from time import time
from utils import export_net

from collections import defaultdict
from time import time
from node import Node
from heap import Heap

class Graph:  
    """ Represents an undirected graph using adjacency list representation """

    def __init__(self, vertices = 0, dist=2, filename=None):
        if filename:
            self.import_net(filename)
        else: 
            self.V = vertices
            self.create_graph()
        self.priority_list = [-1] * self.V
        self.priority = 0
        self.dist = dist

    def create_graph(self):
        self.graph = []
        for i in range(self.V):
            self.graph.append(Node(i))

    def compute_ci(self, nodes):
        for node in nodes:
            if node.ci < 0:
                node.ci = self.compute_ci_node(node.index, self.dist)
        print("Finished calculating ci")

    def compute_ci_node(self, start_node, dist):
        ci = 0
        visited = [0] * self.V
        visited[start_node] = 1 
        deg = 0
        counters = [0] * (dist+2+1)
        edges = [0] * (dist+2+1)
        edges[0] = self.graph[start_node].degree
        counters[0] = self.graph[start_node].degree
        queue = [start_node]
        bubble = []
        border = []
        count_nodes = [0] * (dist+1+1)
        while len(queue) != 0:
            node = queue.pop(0)
            for adj_node in self.graph[node].neighbors:
                idx_adj_node = adj_node.index
                # print("adj_node: ", adj_node)
                counters[deg] -= 1
                # if not visited
                if visited[idx_adj_node] == 0:
                    # mark as visited, and add to queue
                    count_nodes[deg] += 1
                    visited[idx_adj_node] = 1
                    queue.append(idx_adj_node)
                    # borders[deg].append(adj_node)
                    if deg == (dist+1):
                        border.append(adj_node)
                    else:
                        bubble.append(adj_node)
                    counters[deg+1] += self.graph[idx_adj_node].degree
                    edges[deg+1] += self.graph[idx_adj_node].degree
            if counters[deg] == 0:
                deg += 1
                if deg == (dist+1):
                    ci = (edges[0]-1)*(edges[deg] - count_nodes[deg-1])
                    # print("edges: ", edges)
                    # print("borders: ", borders)
                    # print("count_nodes: ", count_nodes)
                if deg == (dist+2):
                    self.graph[start_node].border = border
                    self.graph[start_node].bubble = bubble
                    return ci
        return ci

    def update_ci_values(self, center_node):
        decrease_amnt = center_node.degree - 1
        for node in center_node.border:
            node.ci -= decrease_amnt
        self.compute_ci(center_node.bubble)

    def create_max_heap(self):
        # heap index starts at 1
        self.heap = Heap([Node(-1)] + self.graph)
        self.heap.build_max_heap()

    def remove_largest_value(self):
        node = self.heap.heap_extract_max()
        if not node:
            import pdb;pdb.set_trace()
        node.kill()

        return node

    def import_net(self, filename):

        print(filename)
        mt = np.genfromtxt('networks/' + filename + '.csv',delimiter=',').astype(np.int32)
        
        # initialize V
        max_node = np.max(mt)
        self.V = max_node+1
        self.create_graph()

        # create adjacency list
        for row in mt[:1000]:
            node1 = row[0]
            node2 = row[1]
            self.addEdge(self.graph[node1], self.graph[node2])

        # assert (self.V == len(self.graph), "Numero de nos incorreto"
        print("Adicionado todas as arestas ao grafo")

    # function to add an edge to graph
    def addEdge(self,u,v):
        if type(u) == int and type(v) == int:
            u = self.graph[u]
            v = self.graph[v]
        u.add_neighbor(v)
        v.add_neighbor(u)

    def run(self):
        self.compute_ci(self.graph)
        self.create_max_heap()
        # counter = self.V-1
        i = 0
        while True:
            if i > int(self.V*0.1):
                break
            center_node = self.remove_largest_value()
            self.priority_list[center_node.index] = self.priority
            self.update_ci_values(center_node)
            # counter -= 1
            i +=1 
            self.priority += 1
            # print("Remaining iterations: {}".format(counter))
        print("Finished running")

    def score_remaining(self, base_score=0.5):
        """  Calculate score for remaining nodes """

        X = [(node.index, node.degree) for node in self.graph]
        X = sorted(X, key=lambda x:-x[1])
        for node, _ in X:
            if priority_list[node] == -1:
                priority_list[node] = self.priority
                self.priority += 1


    def export(self):

        to_export = zip(range(self.V), self.priority_list)
        to_export = sorted(list(to_export), key=lambda x:x[1])
        return to_export

# g = Graph(9, dist=0)
# g.addEdge(0,1)
# g.addEdge(0,2)
# g.addEdge(1,2)
# g.addEdge(1,5)
# g.addEdge(2,3)
# g.addEdge(2,4)
# g.addEdge(2,6)
# g.addEdge(2,5)
# g.addEdge(5,6)
# g.addEdge(5,8)
# g.addEdge(6,8)
# g.addEdge(6,7)
# g.addEdge(6,4)
# g.addEdge(6,3)
# g.addEdge(3,7)
# g.addEdge(3,4)
# g.addEdge(4,7)

# g.run()
# print(g.priority_list)
# for node in g.graph:
#    print(node)
# print(g.export())

network_name = 'real2'
file_out = "ci_{}.csv".format(network_name)
g = Graph(filename=network_name)
t0 = time()
g.run()
print("Running time is {}".format(time()-t0))

export_net(g.export(), network_name, file_out, first=True)

