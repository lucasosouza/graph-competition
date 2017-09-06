
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from pprint import pprint
import math
from utils import import_net, export_net
from time import time


# In[ ]:

def sort_nodes(t):
    v = t[1]
    return -v[0], -v[1]

def BFS(graph, start):
    """ André, please clarify:

        this code is adding +1 to the second counter
        if the node connected by an outgoing edge
        also has an outgoing edge going back to the parent node
        
        is this what we were looking for?
        my understanding is that we wanted to count the number of incoming nodes,
        regardless of its origin
    """
    
    queue = list()
    output = dict()
    visited = set()
    queue.append(start)
    while queue:
        node = queue.pop(0)
        visited.add(node)
        if node in output:
            output[node][0] = len(graph[node])
        else:
            output[node] = [len(graph[node]), 0]
        for adjacent in graph[node]:
            # check logic of next two lines
            if adjacent in output:
                    output[adjacent][1] += 1;
            else:
                output[adjacent] = [0, 1];
            if adjacent not in visited:
                visited.add(adjacent)
                if adjacent in graph:
                    queue.append(adjacent)
                    
    return sorted(output.items(), key=sort_nodes), visited


# In[ ]:

def sort_forests(f):
    return -len(f)

def get_full_bfs(network):
    """ Run BFS for the entire network """
    
    visited_nodes = set()
    output = list()
    for node in all_nodes:
        if node not in visited_nodes:
            forest, visited = BFS(network, node)
            output.append(forest)
            visited_nodes.update(visited)
            
    return output

def score(full_bfs, num_nodes):
    """ Normalize and calculate score for full bfs"""
    
    output = {}
    for forest in full_bfs:
        forest_size = len(forest)
        for node in forest:
            node_id = node[0]
            num_incoming_nodes = node[1][0]
            num_outgoing_nodes = node[1][1]
            score = (num_incoming_nodes+num_outgoing_nodes) * forest_size / num_nodes / 10e6
            if node_id not in output:
                output[node_id] = score
            else:
                output[node_id] += score
            forest_size -= 1
    
    return sorted(output.items(), key=lambda x:-x[1])
 


# In[ ]:

### ALTERNATIVE: RUN THIS TO TEST
network = { 'A': ['B', 'C'],
            'B': ['A', 'C'],
            'C': ['B', 'D', 'E' ,'F'],
            'D': ['E'],
            'E': ['C'],
            'F': ['B', 'D'],
            'G': ['H', 'I','C'],
            'H': ['G'],
            'I': ['G', 'J']}

all_nodes = ['A', 'B', 'C', 'D', 'E', 
             'F', 'G', 'H', 'I', 'J']

# full_bfs = get_full_bfs(network)
# pprint(full_bfs)
# results = score(full_bfs, len(all_nodes))
# pprint(results)


# In[ ]:

### ON REAL DATA
first = True
output = []
for ftype in ['real','model']:
    for i in range(1,5):
        t0 = time()
        network_name = ftype+str(i)
        network, all_nodes = import_net(network_name)
        full_bfs = get_full_bfs(network)
        results = score(full_bfs, len(all_nodes))
        output.append((network_name, results))
        
        print('Nodes in results: ' + str(len(results)))        
        print("Run-time (in minutes): " + str(int((time()-t0)/60)))
        
        if first:
            export_net(results, network_name, 'results-full-v2.csv', first=True)
            first = False
        else:
            export_net(results, network_name, 'results-full-v2.csv', first=False)
            


# In[ ]:




# In[ ]:



