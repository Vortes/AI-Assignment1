class AwesomeHeap:
    def __init__(self) -> None:
        self.heap = []

    def swap(self, a, b):
        temp = self.heap[a]
        self.heap[a] = self.heap[b]
        self.heap[b] = temp

    def heapify(self):
        i = len(self.heap)//2 - 1
        while i >= 0:
            haha = i
            while haha < len(self.heap)//2:
                if (2*haha + 1 < len(self.heap) and self.heap[haha] > self.heap[2*haha + 1]) or (2*haha + 2 < len(self.heap) and self.heap[haha] > self.heap[2*haha + 2]):
                    if 2*haha+2 >= len(self.heap):
                        self.swap(haha, 2*haha+1)
                        haha = 2*haha+1
                    elif 2*haha+2 < len(self.heap) and self.heap[2*haha+1] < self.heap[2*haha+2]:
                        self.swap(haha, 2*haha + 1)
                        haha =2*haha+1
                    else: 
                        self.swap(haha, 2*haha + 2)
                        haha =2*haha+2
                else:
                    break
            i -= 1
        return

    def push(self, tup: tuple): # pushed tuples look like ((priority_tuple), (cell_coords_tuple))
        self.heap.append(tup)
        self.heapify()
        #tup_index = self.heap.index(tup)
        #if tup >= self.heap[(tup_index-1)//2]:
        #    print("tup:",tup)
        #    print("tup_index:", tup_index)
        #    print("self.heap[(tup_index-1)//2]:",self.heap[(tup_index-1)//2])
        #    return
        #else:
        #    self.heapify()
    def pop(self):
        popped_node = self.heap.pop(0)
        self.heapify()
        return popped_node

#def main():
#    a = AwesomeHeap()
#    a.heap = []
#
#    #a.heapify()
#    print(a.heap)
#
#if __name__ == "__main__":
#    main()