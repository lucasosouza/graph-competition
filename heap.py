# heap.py
import math

class Heap():

    def __init__(self, arr):
        self.h = arr
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
        if l <= self.heap_size and self.h[l][1] > self.h[i][1]:
            largest = l
        else:
            largest = i
        if r <= self.heap_size and self.h[r][1] > self.h[largest][1]:
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
            return
        maxx = self.h[1]
        self.h[1] = self.h[self.heap_size]
        self.heap_size -= 1
        self.max_heapify(1)
        return maxx

    def heap_increase_key(self, i, key):
        if key < self.h[i]:
            return
        self.h[i] = key
        while i > 1 and self.h[self.parent(i)][1] < self.h[i][1]:
            self.h[i], self.h[self.parent(i)] = self.h[self.parent(i)], self.h[i]
            i = self.parent(i)

    def max_heap_insert(self, key):
        self.heap_size += 1
        self.h[self.heap_size][1] = -math.inf
        self.heap_increase_key(self.heapsize, key)


# arr = [0,11,10,8,5,9,7,1,4,6,3,5,1,2,3]
# arr = list(zip([0] + list(range(len(arr))),arr))
# heap = Heap(arr)
# heap.build_max_heap()
# print(heap.h)
# maxx = heap.heap_extract_max()
# print(heap.h)
# print(maxx)