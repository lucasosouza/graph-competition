# Algorithm-2017-09-11.py


import pandas as pd
import numpy as np
from pprint import pprint
import math
from utils import import_net, export_net
from time import time


def sort_nodes(t):
    v = t[1]
    return -v[0], -v[1]

def score_func(network):
    return dict([(v[0], len(v[1])) for v in network.items()])

def BFS(graph, start):
    
    queue = list()
    visited = set()
    queue.append(start)
    while queue:
        node = queue.pop(0)
        visited.add(node)
        for adjacent in graph[node]:
            if adjacent not in visited:
                visited.add(adjacent)
                if adjacent in graph:
                    queue.append(adjacent)
                    
    return list(visited)

# retornar o set de visitados
# pegar o score
# edges

def sort_forests(f):
    return -len(f)

def alg(network):
    """ Run BFS for the entire network """
    
    num_nodes = len(network.keys())
    output = {}
    visited_nodes = set()
    score_table = score_func(network)
    for node in network.keys():
        if node not in visited_nodes:

            # busco a floresta (e atualizo lista de nós visitados)
            forest = BFS(network, node)
            visited_nodes.update(forest)

            # ordenar a floresta pelo score de cada nó
            forest = [(node, score_table[node]) for node in forest]
            forest = sorted(forest, key= lambda x:-x[1])  

            # calcular score
            forest_size = len(forest)
            for node, score in forest:
                score = score * (forest_size / num_nodes)
                output[node] = score
                forest_size -= 1
            
    return sorted(output.items(), key=lambda x:-x[1]) 


### ALTERNATIVE: RUN THIS TO TEST
# network = { 'A': ['B', 'C'],
#             'B': ['A', 'C'],
#             'C': ['B', 'A', 'D'],
#             'D': ['C'],
#             'E': ['F', 'G'],
#             'F': ['E', 'G'],
#             'G': ['F', 'H','E', 'I'],
#             'H': ['G'],
#             'I': ['G']}

# results = alg(network)
# pprint(results)

## ON REAL DATA
first = True
for ftype in ['real', 'model']:
    for i in range(1,5):
        t0 = time()
        network_name = ftype+str(i)
        network = import_net(network_name)
        results = alg(network)
        
        print('Nodes in results: ' + str(len(results)))        
        print("Run-time (in minutes): " + str(int((time()-t0)/60)))
        
        if first:
            export_net(results, network_name, 'results-final.csv', first=True)
            first = False
        else:
            export_net(results, network_name, 'results-final.csv', first=False)

            
## ON REAL DATA
# t0 = time()
# network_name = 'real2'
# network = import_net(network_name)
# results = alg(network)

# print('Nodes in results: ' + str(len(results)))        
# print("Run-time (in minutes): " + str(int((time()-t0)/60)))

# import pdb;pdb.set_trace()
        
            


