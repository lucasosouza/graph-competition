graph = { 'A': ['B', 'C'],
          'B': ['A', 'C'],
          'C': ['B', 'D', 'E' ,'F'],
          'D': ['E'],
          'E': ['C'],
          'F': ['B', 'D'],
          'G': ['H'],
          'H': ['G']}
          
def BFS(graph, start):
    queue = list()
    most = dict()
    visited = set()
    queue.append(start)
    while queue:
        node = queue.pop(0)
        visited.add(node)
        if node in most:
            most[node][0] = len(graph[node])
        else:
            most[node] = [len(graph[node]), 0];
        for adjacent in graph[node]:
            #LOGICA DO QUE A GENTE QUER
            if adjacent in most:
                    most[adjacent][1] += 1;
            else:
                most[adjacent] = [0, 1];
            if adjacent not in visited:
                visited.add(adjacent)
                if adjacent in graph:
                    queue.append(adjacent)
                
    # print(most)
    #SORT MOST EM [0] e depois em [1]
    
    def sort_nodes(t):
        v = t[1]
        return -v[0], -v[1]

    # sorted(d.items(), key=lambda (y,x): (-x[0], -x[1]))
    return sorted(most.items(), key=sort_nodes), visited

def sort_forests(f):
    return -len(f)

real_visited = set()
real_result = list()
for node in graph:
  if node not in real_visited:
    result, visited = BFS(graph, node)
    real_result.append(result)
    real_visited.update(visited)


print(sorted(real_result, key=sort_forests))
print([len(a) for a in real_result])
# print(real_visited)

