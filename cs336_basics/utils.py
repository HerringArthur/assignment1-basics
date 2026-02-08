from typing import Tuple

class Node:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.prev = None
        self.next = None



class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    

    def append(self, value):
        new_node = Node(value)

        if self.head is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

        self.size += 1


    def prepend(self, value):
        new_node = Node(value)

        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

        self.size += 1


    def remove(self, value):
        curr = self.head

        while curr:
            if curr.value == value:
                # 处理前驱
                if curr.prev:
                    curr.prev.next = curr.next
                else:
                    self.head = curr.next

                # 处理后继
                if curr.next:
                    curr.next.prev = curr.prev
                else:
                    self.tail = curr.prev

                self.size -= 1
                return True

            curr = curr.next

        return False



class IndexedMaxHeap:
    def __init__(self):
        self.data = []      # [(value, key)]
        self.pos = {}       # key -> index

    # ---------- 工具 ----------
    def _parent(self, i):
        return (i - 1) // 2

    def _left(self, i):
        return 2 * i + 1

    def _right(self, i):
        return 2 * i + 2

    def _swap(self, i, j):
        self.data[i], self.data[j] = self.data[j], self.data[i]
        self.pos[self.data[i][1]] = i
        self.pos[self.data[j][1]] = j

    # ---------- 核心 ----------
    def push(self, key, value):
        if key in self.pos:
            i = self.pos[key]
            old_val = self.data[i][0]
            new_val = old_val + value
            self.update(key, new_val)
        else:
            self.data.append((value, key))
            idx = len(self.data) - 1
            self.pos[key] = idx
            self._sift_up(idx)

    def pop(self):
        if not self.data:
            raise IndexError("pop from empty heap")

        value, key = self.data[0]
        last = self.data.pop()
        del self.pos[key]

        if self.data:
            self.data[0] = last
            self.pos[last[1]] = 0
            self._sift_down(0)

        return key, value

    def update(self, key, new_value):
        if key not in self.pos:
            raise KeyError("key not found")

        i = self.pos[key]
        old_value, _ = self.data[i]
        self.data[i] = (new_value, key)

        # 决定上浮还是下沉
        if new_value > old_value:
            self._sift_up(i)
        else:
            self._sift_down(i)

    # ---------- 调整 ----------
    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self.data[p][0] >= self.data[i][0]:
                break
            self._swap(i, p)
            i = p

    def _sift_down(self, i):
        n = len(self.data)
        while True:
            l = self._left(i)
            r = self._right(i)
            largest = i

            if l < n and self.data[l][0] > self.data[largest][0]:
                largest = l
            if r < n and self.data[r][0] > self.data[largest][0]:
                largest = r

            if largest == i:
                break

            self._swap(i, largest)
            i = largest

    # ---------- 辅助 ----------
    def contains(self, key):
        return key in self.pos
