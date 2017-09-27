class Node:
    def __init__(self, index):
        self.index = index
        self.ci = -1
        self.bubble = list()
        self.border = list()
        self.degree = 0
        self.neighbors = list()
        self.is_active = True

    def add_neighbor(self,node):
        self.neighbors.append(node)
        self.degree += 1

    def kill(self):
        self.is_active = False
        for adj_node in self.neighbors:
            adj_node.neighbors.remove(self)

    def __str__(self):
        return "{}, {}, {}, {}".format(self.index, self.ci, self.bubble, self.border)
