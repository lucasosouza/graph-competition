import numpy as np
import sys
import math
from pprint import pprint
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

    def Enhanced_Collective_Influence(self, d=2):
    #强化的Collective Influence, 参数d为考虑的范围radius。
        Enhanced_Collective_Influence_Dic = {}
        ECI = 0

        node_set = self.graph
        #对于网路中每个节点。
        for nid in node_set:
            neighbor_hop_1 = nid.neighbors
            neighbor_hop_2 = []
            for ngb1 in neighbor_hop_1:
                neighbor_hop_2 = list(set(neighbor_hop_2).union(ngb1.neighbors))
            #end for
            neighbor_hop_2 = list(  set(neighbor_hop_2).difference( set(neighbor_hop_1).union(set([nid]))  ) )

            #(1)计算Collective_Influence取值
            Total_Reduced_Degree = 0.0
            for id in neighbor_hop_2:
                Total_Reduced_Degree = Total_Reduced_Degree + (id.degree-1.0)
            #end
            Collective_Influence = (nid.degree-1.0) * Total_Reduced_Degree

            #(2)对nid的Collective_Influence进行关于neighbors的Correlation_Intensity强化

            Correlation_Intensity = 0.0

            for id1 in neighbor_hop_2: #Center_set：离中心源点不同层的节点集合
                for id2 in neighbor_hop_2:
                    if id1.index != id2.index:
                        Correlation_Intensity = Correlation_Intensity + float(len(set(id1.neighbors).intersection(set(id2.neighbors)))) / float(len(set(id1.neighbors).union(set(id2.neighbors))))
            #end for

            Correlation_Intensity_1 = 0.0
            for id1 in neighbor_hop_1: #Center_set：离中心源点不同层的节点集合
                for id2 in neighbor_hop_1:
                    if id1.index != id2.index:
                        Correlation_Intensity_1 = Correlation_Intensity_1 + float(len(set(id1.neighbors).intersection( set(id2.neighbors).difference(set([nid]))  ))) / float(len(set(id1.neighbors).union(  set(id2.neighbors).difference(set([nid]))   )))
            #end for
            Correlation_Intensity = 0.5*Correlation_Intensity + Correlation_Intensity_1

            '''
            #SubG_1 = G.subgraph(neighbor_hop_1).copy() #子图
            SubG_2 = G.subgraph(neighbor_hop_2).copy() #子图
            #SubEdge_1 = SubG_1.number_of_edges()
            SubEdge_2 = SubG_2.number_of_edges()
            #SubDegree_1 = sum(G.degree(v) for v in SubG_1.nodes())
            SubDegree_2 = sum(G.degree(v) for v in SubG_2.nodes())
            #Correlation_Intensity = 2*float(SubEdge_1)/(SubDegree_1+1) + float(SubEdge_2)/(SubDegree_2+1)
            Correlation_Intensity = Correlation_Intensity + float(SubEdge_2)/(SubDegree_2+1)
            '''


            #(3)计算邻居结构的均衡性-structural entropy
            #邻居节点的度概率-Degree proporational list
            Degree_List = []
            Total_Degree = 0
            for node in nid.neighbors:
                Degree_List.append(node.degree)
                Total_Degree = Total_Degree + node.degree
            #end for
            for i in range(0,len(Degree_List)):
                Degree_List[i] = Degree_List[i]/float(Total_Degree)
            #end for
            #计算正则化熵
            Entropy = 0.0
            for i in range(0, len(Degree_List)):
                Entropy = Entropy + ( - Degree_List[i] * math.log( Degree_List[i] ) )
            Entropy = Entropy / math.log( nid.degree + 0.1 )
            #end for

            #（4）计算Enhanced_Collective_Influence(ECI)
            ECI = Collective_Influence * Entropy/(1+Correlation_Intensity)
            Enhanced_Collective_Influence_Dic[nid.index] = ECI
            if(VERBOSE):
                print("{}/{}=>ECI:{}".format(nid.index,self.V,ECI))  

        #end for

        #print sorted(Enhanced_Collective_Influence_Dic.iteritems(), key=lambda d:d[1], reverse = True)
        return Enhanced_Collective_Influence_Dic

    def Collective_Influence(self, dist=2):
        Collective_Influence_Dic = {}
        for node in self.graph:
            CI = 0
            neighbor_set = []
            neighbor_hop_1 = node.neighbors
            neighbor_hop_2 = []
            for nnode in neighbor_hop_1:
                neighbor_hop_2  = list(set(neighbor_hop_2).union(set(nnode.neighbors)))
                #print '2_hop:', nnid, G.neighbors(nnid)
            #end for

            center = [node]
            neighbor_set = list(   set(neighbor_hop_2).difference(   set(neighbor_hop_1).union(set(center))  )    )
            #print nid, neighbor_hop_1, neighbor_hop_2, neighbor_set

            total_reduced_degree = 0
            for nnnode in neighbor_set:
                total_reduced_degree = total_reduced_degree + (nnnode.degree-1.0)
            #end
            CI = (node.degree-1.0) * total_reduced_degree
            if(VERBOSE):
                print("{}/{}=>CI:{}".format(node.index,self.V,CI))
            Collective_Influence_Dic[node.index] = CI
        #end for
        #print "Collective_Influence_Dic:",sorted(Collective_Influence_Dic.iteritems(), key=lambda d:d[1], reverse = True)

        return Collective_Influence_Dic

    def compute_ci(self, nodes):
        index = 0
        tot = len(nodes)
        mod = 1000 if len(nodes) > 1000 else len(nodes)
        for node in nodes:
            if VERBOSE:
                if not index%mod:
                    print("CURRENT: {}/{}".format(index,tot))
            # if TYPE:
            #     node.ci = self.compute_ci_node(node.index, self.dist-1)
            # else:
            #     node.ci = self.compute_ci_node_v2(node.index, self.dist)
            if node.is_active:
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
            if DEBUG2:
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
                for node in camada:
                    soma += (node.degree-1.0)
                # soma = (sum(node.degree for node in bola[indice]) - len(bola[indice]))
                break
            indice-=1
        grau = self.graph[start_node].degree -1.0
        ci = grau*soma
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

    # def compute_ci_node(self, start_node, dist):
    #     ci = 0
    #     visited = [0] * self.V
    #     visited[start_node] = 1
    #     deg = 0
    #     counters = [0] * (dist+2+1)
    #     edges = [0] * (dist+2+1)
    #     edges[0] = self.graph[start_node].degree
    #     counters[0] = self.graph[start_node].degree
    #     queue = [start_node]
    #     bubble = []
    #     border = []
    #     count_nodes = [0] * (dist+1+1)
    #     while len(queue) != 0:
    #         node = queue.pop(0)
    #         for adj_node in self.graph[node].neighbors:
    #             idx_adj_node = adj_node.index
    #             # print("adj_node: ", adj_node)
    #             counters[deg] -= 1
    #             # if not visited
    #             if visited[idx_adj_node] == 0:
    #                 # mark as visited, and add to queue
    #                 count_nodes[deg] += 1
    #                 visited[idx_adj_node] = 1
    #                 queue.append(idx_adj_node)
    #                 # borders[deg].append(adj_node)
    #                 if deg == (dist+1):
    #                     border.append(adj_node)
    #                 else:
    #                     bubble.append(adj_node)
    #                 counters[deg+1] += self.graph[idx_adj_node].degree
    #                 edges[deg+1] += self.graph[idx_adj_node].degree
    #         if counters[deg] == 0:
    #             deg += 1
    #             if deg == (dist+1):
    #                 ci = (edges[0]-1)*(edges[deg] - count_nodes[deg-1])
    #                 # print("edges: ", edges)
    #                 # print("borders: ", borders)
    #                 # print("count_nodes: ", count_nodes)
    #             if deg == (dist+2):
    #                 self.graph[start_node].border = border
    #                 self.graph[start_node].bubble = bubble
    #                 return ci
    #     return ci

    def update_ci_values(self, center_node):
        decrease_amnt = center_node.degree - 1
        # if TYPE:
        #     for node in center_node.border:
        #         node.ci -=  decrease_amnt
        #     flat_list = center_node.bubble
        # else:
            # for node in center_node.e_border:
            #     node.ci -= decrease_amnt
            # flat_list = [item for sublist in center_node.bola for item in sublist]
            # index_list = [node.index for node in flat_list]
        if DEBUG:
            print("RECALCULAR CI E_BORDER")
            index_list = [node.index for node in center_node.e_border]
            print(index_list)
        for node in center_node.e_border:
            node.ci -= decrease_amnt
        flat_list = [item for sublist in center_node.bola for item in sublist]
        if DEBUG:
            print("BOLA PRECISA SER RECALCULADA")
            index_list = [node.index for node in flat_list]
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
            self.kill(node)
        return node

    def import_net(self, filename):
        print(filename)
        mt = np.genfromtxt('networks/' + filename + '.csv',delimiter=',').astype(np.int32)
        
        # initialize V
        max_node = np.max(mt)
        self.V = max_node+1
        self.create_graph()

        # create adjacency list
        for row in mt:
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

    def kshell(self,current_degree):
        _res = list()
        found = True
        i = 0
        to_kill = []
        while(found):
            j = 0
            _res.append([])
            to_kill = []
            for node in self.graph:
                if DEBUG:
                    print("Current node: {}".format(node.index))
                    print("Neighbors:")
                    for adj in node.neighbors:
                        print(adj.index)
                if node.degree==current_degree and node.is_active:
                    if DEBUG:
                        print("Found node: {}".format(node.index))
                    _res[i].append(node.index)
                    to_kill.append(node)
                    j+=1
            if j == 0:
                found = False
            else:
                for node in to_kill:
                    self.kill(node)
                i+=1
        return _res
    def full_kshell(self):
        res = []
        max_degree = max(node.degree for node in self.graph)
        # print(res)
        # res = self.kshell(1)
        # print(res)
        # res = self.kshell(2)
        # print(res)
        # res = self.kshell(3)
        # print(res)
        # res = self.kshell(2)
        # print(res)
        for i in range(1,int((max_degree/2))+1):
            res.append(self.kshell(i))
        for i in reversed(range(1,int((max_degree/2))+1)):
            res.append(self.kshell(i))
        res = [item for sublist in res for item in sublist]
        res = [item for sublist in res for item in sublist]
        # print (list(reversed(res)))
        return list(reversed(res))
    def kill(self,node):
        if DEBUG:
            print("KILLED NODE: {}".format(node.index))
        node.kill()

    def run_eci(self):
        res = self.Enhanced_Collective_Influence()
        res = sorted(res.items(), key=lambda d:-d[1])
        return res
    def run(self):
        res = self.Collective_Influence()
        res = sorted(res.items(), key=lambda d:-d[1])
        return res
    def run_ci(self):
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
global DEBUG2
global TYPE
global VERBOSE
if __name__ == "__main__":
    DEBUG = 0
    DEBUG2 = 0
    TYPE = 0
    VERBOSE = 0
    for arg in sys.argv:
        if arg == "--debug":
            DEBUG = 1
        if arg == "--debug2":
            DEBUG = 2
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

    # print("TAMANHO DO GRAFO: {}".format(g.V))
    # res = g.full_kshell()
    # print(res)
    # res = g.run()
    # print(res)
    # res = g.run_eci()
    # pprint(res)
    # g.run_ci()
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
    # network_type = ['model', 'real']
    # for type in network_type:
        # for i in range(0,4):
            # network_name = "{}{}".format(type,i+1)
    network_name = "model3"
    file_out = "eci_{}_{}.csv".format(network_name,_dist)
    # load graph
    if(VERBOSE):
        print("Rodando {}".format(network_name))
        print("Saida em {}".format(file_out))
    g = Graph(filename=network_name, dist=_dist)
    # t0 = time()
    # # run
    print("Calculando ECI")
    # res = g.run()
    res = g.run_eci()
    print(res[:1000])
    export_net(res,network_name,file_out,first=True)
    # if(VERBOSE):
    #     print("Running time is {}".format(time()-t0))
    # res = g.export()
    # for node in res[:1000]:
    #     print(node)
            # save full file as pickle
            # with open('{}.p'.format(network_name), 'wb') as f:
            #     pickle.dump(res, f)

            # export in contest format
            # export_net(res, network_name, file_out, first=True)
