import numpy as np
import sys
from time import time
from utils import export_net

from collections import defaultdict
from time import time
from node import Node
from heap import Heap

import pickle

class Graph:  
    """ Represents an undirected graph using adjacency list representation """

    def __init__(self, vertices = 0, dist=2, filename=None):
        if filename:
            self.import_net(filename)
        else: 
            self.V = vertices
            self.create_graph()
        # self.priority_list = [-1] * self.V
        # self.priority = 0
        self.dist = dist

    def create_graph(self):
        self.graph = []
        for i in range(self.V):
            self.graph.append(Node(i))

    def compute_ci(self, nodes):
        index = 0
        tot = len(nodes)
        mod = 1000 if len(nodes) > 1000 else len(nodes)
        for node in nodes:
            if VERBOSE:
                if not index%mod:
                    print("{}/{}".format(index,tot))
            if TYPE:
                node.ci = self.compute_ci_node(node.index, self.dist-1)
            else:
                node.ci = self.compute_ci_node_v2(node.index, self.dist)
            index+=1
        if DEBUG:
            print("CI PARA LISTA REQUERIDA CALCULADO")

    def compute_ci_node_v2(self, start_node, dist):
        ci = 0
        level = 0
        visited = [0]*self.V
        visited[start_node] = 1
        queue = [start_node]
        queue.append(None)
        bola = []
        bola_idx = []
        if DEBUG:
            print("START BFS: {}".format(start_node))
        for i in range(0,dist):
            bola.append([])
            bola_idx.append([])
        e_border = []
        e_border_idx = []
        while len(queue) != 0:
            node = queue.pop(0)
            if DEBUG:
                print("node: {}, level: {}".format(node, level))
                print(visited)
            if node is None:
                level+=1
                if(level==dist+1):
                    if(DEBUG):
                        print("Chegou ao nivel requerido")
                    break
                queue.append(None)
                if queue[0] == None:
                    break
                else:
                    continue
            for adj_node in self.graph[node].neighbors:
                idx_adj_node = adj_node.index
                if not visited[idx_adj_node]:
                    visited[idx_adj_node] = 1
                    queue.append(idx_adj_node)
                    if level!=dist:
                        bola[level].append(adj_node)
                        bola_idx[level].append(idx_adj_node) #LUCAS VE ISSO AQUI
                    else:
                        e_border.append(adj_node)
                        e_border_idx.append(idx_adj_node)
        # ci = (self.graph[start_node].degree - 1)*(sum(x[1] for x in bola[-1]) - len(bola[-1]))
        indice = len(bola) -1
        soma = 0
        for camada in reversed(bola):
            if len(camada)!=0:
                soma = (sum(node.degree for node in bola[indice]) - len(bola[indice]))
                break
            indice-=1
        ci = (self.graph[start_node].degree - 1)*soma
        self.graph[start_node].bola = bola
        self.graph[start_node].e_border = e_border
        if DEBUG:
            flat_list = [item for sublist in bola for item in sublist]
            print("BOLA: {}".format(bola_idx))
            index_list = [node.index for node in e_border]
            print("E_BORDER: {}".format(e_border_idx))
            print("CI de {}: {}".format(start_node,ci))
            input()
        return ci

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
        if TYPE:
            for node in center_node.border:
                node.ci -=  decrease_amnt
            flat_list = center_node.bubble
        else:
            for node in center_node.e_border:
                node.ci -= decrease_amnt
            flat_list = [item for sublist in center_node.bola for item in sublist]
            index_list = [node.index for node in flat_list]
        if DEBUG:
            print("CENTER NODE: {}".format(center_node.index))
            print("BOLA PRECISA SER RECALCULADA")
            print(index_list)
            input()
        self.compute_ci(flat_list)

    def create_max_heap(self):
        # heap index starts at 1
        self.heap = Heap([Node(-1)] + self.graph)
        self.heap.build_max_heap()

    def remove_largest_value(self):
        node = self.heap.heap_extract_max()
        if isinstance(node, Node):        
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
        if VERBOSE:
            print("ADD TODOS EDGES")

    # function to add an edge to graph
    def addEdge(self,u,v):
        if isinstance(u,int) and isinstance(v,int):
            u = self.graph[u]
            v = self.graph[v]
        u.add_neighbor(v)
        v.add_neighbor(u)

    def run(self):
        self.compute_ci(self.graph)
        self.create_max_heap()
        i = 0
        stop_condition = int(self.V*1)
        while True:
            if i > stop_condition:
                break
            center_node = self.remove_largest_value()
            if isinstance(center_node, Node):
                # self.priority_list[center_node.index] = self.priority
                self.update_ci_values(center_node)
                # self.priority += 1
            i +=1 
            # print("Remaining iterations: {}".format(counter))

        print("TERMINOU A EXECUÇÃO")

    # def score_remaining(self, base_score=0.5):
    #     """  Calculate score for remaining nodes """

    #     X = [(node.index, node.ci) for node in self.graph]
    #     X = sorted(X, key=lambda x:-x[1])
    #     for node, _ in X:
    #         if self.priority_list[node] == -1:
    #             self.priority_list[node] = self.priority
    #             self.priority += 1


    def export(self):
        
        to_export = []
        # for idx, priority in enumerate(self.priority_list):
        #     to_export.append((idx, priority, self.graph[idx].ci))

        to_export = sorted(self.graph, key=lambda x:-x.ci)
        return to_export

global DEBUG
global TYPE
global VERBOSE
if __name__ == "__main__":
    DEBUG = 0
    TYPE = 0
    VERBOSE = 0
    for arg in sys.argv:
        if arg == "--debug":
            DEBUG = 1
        if arg == "--og":
            TYPE = 1
        if arg == "--verbose":
            VERBOSE = 1
    _dist = 2
    # print("GRAFO EXEMPLO K-CORE")
    # g = Graph(9, _dist)
    # print("LARGURA BOLA: {}".format(g.dist))
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

    # print(g.V)
    # g.run()
    # # _graph = sorted(g.graph, key=lambda x:-x.ci)
    # # for node in _graph:
    # #     print(node)
    # res = g.export()
    # for node in res:
    #     print(node)
    # export_net(res,"teste","teste.csv",first=True)

    # print("GRAFO2 EXEMPLO 3-CORE")
    # g = Graph(5, dist = 1)
    # g.addEdge(0,1)
    # g.addEdge(0,2)
    # g.addEdge(0,3)
    # g.addEdge(0,4)
    # g.addEdge(1,2)
    # g.addEdge(1,4)
    # g.addEdge(2,3)
    # g.addEdge(2,4)
    # g.addEdge(3,4)
    # network_name = 'real5'

    # define file in and file out
    network_type = ['model', 'real']
    for type in network_type:
        for i in range(0,4):
            network_name = "{}{}".format(type,i+1)
            file_out = "./Results/ci_{}_{}.csv".format(network_name,_dist)
	        # load graph
            if(VERBOSE):
                print("Rodando {}".format(network_name))
                print("Saida em {}".format(file_out))
            g = Graph(filename=network_name, dist=_dist)
            t0 = time()
            # run
            g.run()
            if(VERBOSE):
                print("Running time is {}".format(time()-t0))
            res = g.export()

            # save full file as pickle
            # with open('{}.p'.format(network_name), 'wb') as f:
            #     pickle.dump(res, f)

            # export in contest format
            export_net(res, network_name, file_out, first=True)
