# heap.py
import math
from node import Node

class Heap():

    def __init__(self, nodes):
        self.h = nodes
        self.heap_size = len(self.h) - 1

    def parent(self, i):
        return math.floor(i/2)

    def left(self, i):
        return 2*i

    def right(self,i):
        return 2*i+1

    def max_heapify(self, i):
        l = self.left(i)
        r = self.right(i)
        if l <= self.heap_size and self.h[l].ci > self.h[i].ci:
            largest = l
        else:
            largest = i
        if r <= self.heap_size and self.h[r].ci > self.h[largest].ci:
            largest = r
        if largest != i:
            self.h[i], self.h[largest] = self.h[largest], self.h[i]
            self.max_heapify(largest)

    def build_max_heap(self):
        self.heap_size = len(self.h)-1
        for i in range(math.floor(self.heap_size/2),0,-1):
            self.max_heapify(i)

    def heapsort(self):
        self.build_max_heap()
        for i in range(len(self.h)-1,1,-1):
            self.h[1], self.h[i] = self.h[i], self.h[1]
            self.heap_size -= 1
            self.max_heapify(1)

    def heap_maximum(self):
        return self.h[1]

    def heap_extract_max(self):
        if self.heap_size < 1:
            return False
        maxx = self.h[1]
        self.h[1] = self.h[self.heap_size]
        self.heap_size -= 1
        self.max_heapify(1)
        return maxx

    def heap_increase_key(self, i, key):
        if key < self.h[i]:
            return
        self.h[i] = key
        while i > 1 and self.h[self.parent(i)].ci < self.h[i].ci:
            self.h[i], self.h[self.parent(i)] = self.h[self.parent(i)], self.h[i]
            i = self.parent(i)

    def max_heap_insert(self, key):
        self.heap_size += 1
        self.h[self.heap_size].ci = -math.inf
        self.heap_increase_key(self.heapsize, key)

    def print_heap(self):
        for node in self.h:
            print(node)


# arr = [0,11,10,8,5,9,7,1,4,6,3,5,1,2,3]
# nodes = []
# for i, ci in enumerate(arr):
#     node = Node(i-1)
#     node.ci = ci
#     nodes.append(node)

# heap = Heap(nodes)
# heap.build_max_heap()
# heap.print_heap()
# print("\n\n")
# maxx = heap.heap_extract_max()
# heap.print_heap()
# print(maxx)