import numpy as np

def import_net(file_name):
    """ Import CSV file in the competition format
        Output graph as adjacency list and a set with all nodes """

    print(file_name)
    mt = np.genfromtxt('networks/' + file_name + '.csv',delimiter=',').astype(np.int32)

    # create adjacency list
    network = {}
    for row in mt:
        source = row[0]
        target = row[1]
        if source not in network:
            network[source] = set()
        network[source].add(target)
        
    print("Nodes with outgoing edges: ", len(network))

    # create set of nods, including nodes without outgoing edges
    all_nodes = set(mt.flatten())
    print("All nodes: ", len(all_nodes))

    return network, all_nodes


def export_net(results, network_name, file_out, first=True):
    """ Export to format required by competition 
        Required: optimize/clean last if, redundancy """

    if first:
        f = open(file_out, 'w')

        # write headers
        f.write('NetID,')
        for i in range(1, 500):
            f.write('nodeID' + str(i) + ',')
        f.write('nodeID500')

    else:
        f = open(file_out, 'a')

    f.write('\n')
    # write each row
    counter = 1
    for node, score in results:
        if counter == 1:
            # add column header when counter is 1
            f.write(network_name + ',')
            counter += 1
        if counter == 501:
            f.write(str(node))
            f.write('\n')
            # reset counter at 500
            counter = 1
        else:
            f.write(str(node) + ',')
            counter += 1
            
    # calculate how much padding is required to be added
    padding = 0
    mod = len(results) % 500
    if mod != 0:
        padding = 500 - mod

    # add padding to last row
    for i in range(padding-1):
        f.write(',')

    f.close()


    print("Network " + network_name + " exported successfully.")



####### Single export version of export_net


# def export_net(output, file_out):
#     """ Export to format required by competition 
#         Required: optimize/clean last if, redundancy 
#         Version optimized for computation, not memory"""

#     f = open(file_out, 'w')

#     # write headers
#     f.write('NetID,')
#     for i in range(1, 500):
#         f.write('nodeID' + str(i) + ',')
#     f.write('nodeID500')

#     for network_name, results in output:
        
#         # move to next line
#         f.write('\n')

#         # write each row
#         counter = 1
#         for node, score in results:
#             if counter == 1:
#                 # add column header when counter is 1
#                 f.write(network_name + ',')
#                 counter += 1
#             if counter == 501:
#                 f.write(str(node))
#                 f.write('\n')
#                 # reset counter at 500
#                 counter = 1
#             else:
#                 f.write(str(node) + ',')
#                 counter += 1
                
#         # calculate how much padding is required to be added
#         padding = 0
#         mod = len(results) % 500
#         if mod != 0:
#             padding = 500 - mod

#         # add padding to last row
#         for i in range(padding-1):
#             f.write(',')

#         print("Network " + network_name + " exported successfully.")

#     f.close()



