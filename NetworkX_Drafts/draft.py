import networkx as nx
from utils import import_net, export_net, exportGML_net
from pylab import show, hist, figure
# G = nx.read_gml('real2.gml',True)

def most_important(G):
 """ returns a copy of G with
     the most important nodes
     according to the pagerank """ 
 ranking = nx.betweenness_centrality(G).items()
 print (ranking)
 r = [x[1] for x in ranking]
 m = sum(r)/len(r) # mean centrality
 t = m*3 # threshold, we keep only the nodes with 3 times the mean
 Gt = G.copy()
 for k, v in ranking:
  if v < t:
   Gt.remove_node(k)
 return Gt

network_name = "model2";
network, all_nodes = import_net(network_name);
exportGML_net(network, all_nodes, network_name + ".gml");
# full_bfs = get_full_bfs(network)
# results = score(full_bfs, len(all_nodes))
#output.append((network_name, results))
# Gt = most_important(G) # trimming

# # create the layout
# pos = nx.spring_layout(G)
# # draw the nodes and the edges (all)
# nx.draw_networkx_nodes(G,pos,node_color='b',alpha=0.2,node_size=8)
# nx.draw_networkx_edges(G,pos,alpha=0.1)

# # draw the most important nodes with a different style
# nx.draw_networkx_nodes(Gt,pos,node_color='r',alpha=0.4,node_size=254)
# # also the labels this time
# nx.draw_networkx_labels(Gt,pos,font_size=12,font_color='b')
# show()