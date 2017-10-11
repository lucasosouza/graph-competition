class Node:
    def __init__(self, index):
        self.index = index
        self.label = index
        self.ci = -1
        self.bola_nodes = list()
        self.bola = list()
        self.e_border = list()
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
            adj_node.degree -= 1

    def __str__(self):
        _bubble = list()
        _border = list()
        flat_list = [item for sublist in self.bola for item in sublist]
        for node in flat_list:
            _bubble.append(node.index)
        for node in self.e_border:
            _border.append(node.index)
        return "({}, {})".format(self.index, self.ci)
