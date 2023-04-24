### Variables ###
# Variables are dynamicly typed
n = 0
print('n =', n)
>>> n = 0

n = "abc"
print('n =', n)
>>> n = abc

# Multiple assignments
n, m = 0, "abc"
n, m, z = 0.125, "abc", False

# Increment
n = n + 1 # good
n += 1    # good
n++       # bad

# None is null (absence of value)
n = 4
n = None
print("n =", n)
>>> n = None
### If-Statements
# If statements don't need parentheses 
# or curly braces.
n = 1
if n > 2:
    n -= 1
elif n == 2:
    n *= 2
else:
    n += 2

# Parentheses needed for multi-line conditions.
# and = &&
# or  = ||
n, m = 1, 2
if ((n > 2 and 
    n != m) or n == m):
    n += 1

### Loops
n = 0
while n < 5:
    print(n)
    n += 1
>>> 0 1 2 3 4

# Looping from i = 0 to i = 4
for i in range(5):
    print(i)
>>> 0 1 2 3 4

# Looping from i = 2 to i = 5
for i in range(2, 6):
    print(i)
>>> 2 3 4 5

# Looping from i = 5 to i = 2
for i in range(5, 1, -1):
    print(i)
>>> 5 4 3 2

### Math
# Division is decimal by default
print(5 / 2)
>>> 2.5

# Double slash rounds down
print(5 // 2)
>>> 2

# CAREFUL: most languages round towards 0 by default
# So negative numbers will round down
print(-3 // 2)
>>> -2

# A workaround for rounding towards zero
# is to use decimal division and then convert to int.
print(int(-3 / 2))
>>> -1

# Modding is similar to most languages
print(10 % 3)
>>> 1

# Except for negative values
print(-10 % 3)
>>> 2

# To be consistent with other languages modulo
import math
from multiprocessing import heap
print(math.fmod(-10, 3))
>>> -1

# More math helpers
print(math.floor(3 / 2))
>>> 1
print(math.ceil(3 / 2))
>>> 2
print(math.sqrt(2))
>>> 1.4142135623730951
print(math.pow(2, 3))
>>> 8

# Max / Min Int
float("inf")
float("-inf")

# Python numbers are infinite so they never overflow
print(math.pow(2, 200))
>>> 1.6069380442589903e+60

# But still less than infinity
print(math.pow(2, 200) < float("inf"))
>>> True

### Arrays
# Arrays (called lists in python)
arr = [1, 2, 3]
print(arr)
>>> [1, 2, 3]

# Can be used as a stack
arr.append(4)
arr.append(5)
print(arr)
>>> [1, 2, 3, 4, 5]

arr.pop()
print(arr)
>>> [1, 2, 3, 4]

arr.insert(1, 7)
print(arr)
>>> [1, 7, 2, 3, 4]

arr[0] = 0
arr[3] = 0
print(arr)
>>> [0, 7, 2, 0, 4]

# Initialize arr of size n with default value of 1
n = 5
arr = [1] * n
print(arr)
>>> [1, 1, 1, 1, 1]
print(len(arr))
>>> 5

# Careful: -1 is not out of bounds, it's the last value
arr = [1, 2, 3]
print(arr[-1])
>>> 3

# Indexing -2 is the second to last value, etc.
print(arr[-2])
>>> 2

# Sublists (aka slicing)
arr = [1, 2, 3, 4]
print(arr[1:3])
>>> [2, 3]

# Similar to for-loop ranges, last index is non-inclusive
print(arr[0:4])
>>> [1, 2, 3, 4]

# But no out of bounds error
print(arr[0:10])
>>> [1, 2, 3, 4]

# Unpacking
a, b, c = [1, 2, 3]
print(a, b, c)
>>> 1, 2, 3

# Be careful though, this throws an error
a, b = [1, 2, 3]

# Looping through arrays
nums = [1, 2, 3]

# Using index
for i in range(len(nums)):
    print(nums[i])
>>> 1 2 3

# Without index
for n in nums:
    print(n)
>>> 1 2 3

# With index and value
for i, n in enumerate(nums):
    print(i, n)
>>> 0 1
>>> 1 2
>>> 2 3

# Loop through multiple arrays simultaneously with unpacking
nums1 = [1, 3, 5]
nums2 = [2, 4, 6]
for n1, n2 in zip(nums1, nums2):
    print(n1, n2)
>>> 1 2
>>> 3 4
>>> 5 6

# Reverse
nums = [1, 2, 3]
nums.reverse()
print(nums)
>>> [3, 2, 1]


# Sorting
arr = [5, 4, 7, 3, 8]
arr.sort()
print(arr)
>>> [3, 4, 5, 7, 8]

arr.sort(reverse=True)
print(arr)
>>> [8, 7, 5, 4, 3]

arr = ["bob", "alice", "jane", "doe"]
arr.sort()
print(arr)
>>> ["alice", "bob", "doe", "jane"]

# Custom sort (by length of string)
arr.sort(key=lambda x: len(x))
print(arr)
>>> ["bob", "doe", "jane", "alice"]

# List comprehension
arr = [i for i in range(5)]
print(arr)
>>> [0, 1, 2, 3, 4]

# 2-D lists
arr = [[0] * 4 for i in range(4)]
print(arr)
print(arr[0][0], arr[3][3])
>>> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# This won't work as you expect it to
arr = [[0] * 4] * 4

### Strings
# Strings are similar to arrays
s = "abc"
print(s[0:2])
>>> "ab"

# But they are immutable, this won't work
s[0] = "A"

# This creates a new string
s += "def"
print(s)
>>> "abcdef"

# Valid numeric strings can be converted
print(int("123") + int("123"))
>>> 246

# And numbers can be converted to strings
print(str(123) + str(123))
>>> "123123"

# In rare cases you may need the ASCII value of a char
print(ord("a"))
print(ord("b"))
>>> 97
>>> 98

# Combine a list of strings (with an empty string delimitor)
strings = ["ab", "cd", "ef"]
print("".join(strings))
>>> "abcdef"

### Queues
# Queues (double ended queue)
from collections import deque

queue = deque()
queue.append(1)
queue.append(2)
print(queue)
>>> deque([1, 2])

queue.popleft()
print(queue)
>>> deque([2])

queue.appendleft(1)
print(queue)
>>> deque([1, 2])

queue.pop()
print(queue)
>>> deque([1])

### HashSets
# HashSet
mySet = set()

mySet.add(1)
mySet.add(2)
print(mySet)
>>> {1, 2}
print(len(mySet))
>>> 2

print(1 in mySet)
>>> True
print(2 in mySet)
>>> True
print(3 in mySet)
>>> False

mySet.remove(2)
print(2 in mySet)
>>> False

# list to set
print(set([1, 2, 3]))
>>> {1, 2, 3}

# Set comprehension
mySet = { i for i in range(5) }
print(mySet)
>>> {0, 1, 2, 3, 4}

### HashMaps
# HashMap (aka dict)
myMap = {}
myMap["alice"] = 88
myMap["bob"] = 77
print(myMap)
>>> {"alice": 88, "bob": 77}

print(len(myMap))
>>> 2

myMap["alice"] = 80
print(myMap["alice"])
>>> 80

print("alice" in myMap)
>>> True

myMap.pop("alice")
print("alice" in myMap)
>>> False

myMap = { "alice": 90, "bob": 70 }
print(myMap)
>>> { "alice": 90, "bob": 70 }

# Dict comprehension
myMap = { i: 2*i for i in range(3) }
print(myMap)
>>> { 0: 0, 1: 2, 2: 4 }

# Looping through maps
myMap = { "alice": 90, "bob": 70 }
for key in myMap:
    print(key, myMap[key])
>>> "alice" 90
>>> "bob" 70

for val in myMap.values():
    print(val)
>>> 90
>>> 70

for key, val in myMap.items():
    print(key, val)
>>> "alice" 90
>>> "bob" 70

### Tuples
# Tuples are like arrays but immutable
tup = (1, 2, 3)
print(tup)
>>> (1, 2, 3)

print(tup[0])
>>> 1

print(tup[-1])
>>> 3

# Can't modify, this won't work
tup[0] = 0

# Can be used as key for hash map/set
myMap = { (1,2): 3 }
print(myMap[(1,2)])
>>> 3

mySet = set()
mySet.add((1, 2))
print((1, 2) in mySet)
>>> True

# Lists can't be keys
myMap[[3, 4]] = 5

### Heaps
import heapq

# under the hood are arrays
minHeap = []
heapq.heappush(minHeap, 3)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 4)

# Min is always at index 0
print(minHeap[0])
>>> 2

while len(minHeap):
    print(heapq.heappop(minHeap))
>>> 2 3 4

# No max heaps by default, work around is
# to use min heap and multiply by -1 when push & pop.
maxHeap = []
heapq.heappush(maxHeap, -3)
heapq.heappush(maxHeap, -2)
heapq.heappush(maxHeap, -4)

# Max is always at index 0
print(-1 * maxHeap[0])
>>> 4

while len(maxHeap):
    print(-1 * heapq.heappop(maxHeap))
>>> 4 3 2

# Build heap from initial values
arr = [2, 1, 8, 4, 5]
heapq.heapify(arr)
while arr:
    print(heapq.heappop(arr))
>>> 1 2 4 5 8

### Functions
def myFunc(n, m):
    return n * m

print(myFunc(3, 4))
>>> 12

# Nested functions have access to outer variables
def outer(a, b):
    c = "c"

    def inner():
        return a + b + c
    return inner()

print(outer("a", "b"))
>>> "abc"

# Can modify objects but not reassign
# unless using nonlocal keyword
def double(arr, val):
    def helper():
        # Modifying array works
        for i, n in enumerate(arr):
            arr[i] *= 2
        
        # will only modify val in the helper scope
        # val *= 2

        # this will modify val outside helper scope
        nonlocal val
        val *= 2
    helper()
    print(arr, val)

nums = [1, 2]
val = 3
double(nums, val)
>>> [2, 4] 6

### Classes
class MyClass:
    # Constructor
    def __init__(self, nums):
        # Create member variables
        self.nums = nums
        self.size = len(nums)
    
    # self key word required as param
    def getLength(self):
        return self.size

    def getDoubleLength(self):
        return 2 * self.getLength()

myObj = MyClass([1, 2, 3])
print(myObj.getLength())
>>> 3
print(myObj.getDoubleLength())
>>> 6


#Array
arr = [3, 2, 1]
arr[0]
arr.append(4)
arr.insert(2, 5)
arr.remove(5)
arr.pop()
arr.reverse()
del arr[::2]

arr[:2]

#Linked Lists
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self) -> None:
        self.head = None
    def get_head(self):
        return self.head
    def is_empty(self):
        return True if self.head is None else False
    def length(self):
        curr = self.head
        length = 0
        while curr:
            length += 1
            curr = curr.next
        return length
    def insert_at_head(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        return self.head
    def insert_at_tail(self, data):
        new_node = Node(data)
        if self.get_head() is None:
            self.head = new_node
            return
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node
        return
    def search(self, val):
        curr = self.get_head()
        while curr:
            if curr.data == val:
                return True
            curr = curr.next
        return False
    def search_recur(self, node, val):
        if (not node):
            return False
        if (node.data is val):
            return True
        return search_recur(self, node.next, val) # can't inside?
    def insert_node(self, prey_node, data):
        if not prey_node:
            print('Previous node does not exist')
        new_node = Node(data)
        new_node.next = prey_node.next
        prey_node.next = new_node
    def delete_at_head(self):
        head = self.get_head()
        if head is not None:
            self.head = head.next
            head = None
        return
    def delete(self, val):
        deleted = False
        if self.is_empty():
            print('List is empty')
            return deleted
        curr = self.get_head()
        prev = None
        if curr.data == val:
            self.delete_at_head()
            deleted = True
            return deleted
        while curr is not None:
            if curr.data == val:
                prev.next = curr.next
                curr.next = None
                deleted = True
                break
            prev = curr
            curr = curr.next
        if deleted == False:
            print(str(val) + ' is not in list')
        else:
            print(str(val) + ' is deleted')
    def delete_node(self, pos):
        if self.head:
            curr = self.head
            if pos == 0:
                self.head = curr.next
                curr = None
                return
            prev = None
            count = 0
            while curr and count != pos:
                prev = curr
                curr = curr.next
                count += 1
            if curr is None:
                return
            prev.next = curr.next
            curr = None
    def reverse(self):
        prev = None
        curr = self.get_head()
        next = None

        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next 
        self.head = prev
    def detect_loop(self):
        slo = self.get_head()
        fas = self.get_head()
        while slo and fas and fas.next:
            slo = slo.next
            fas = fas.next.next
            if slo == fas:
                return True
        return False
    def find_mid(self):
        if self.is_empty():
            return -1
        curr = self.get_head()
        if curr.next == None:
            return curr.data
        mid = curr
        curr = curr.next.next
        while curr:
            mid = mid.next
            curr = curr.next
            if curr:
                curr = curr.next
        if mid:
            return mid.data
        return -1
    def union(self, list):
        if (self.is_empty()):
            return list
        elif (list.is_empty()):
            return self.head
        start = self.get_head()
        while start.next:
            start = start.next
        start.next = list.get_head()
        return self.head

    def find_nth(self, n):
        if (self.is_empty()):
            return -1
        nth_node = self.get_head()
        end_node = self.get_head()
        count = 0
        while count < n:
            if end_node is None:
                return -1
            end_node = end_node.next
            count += 1
        while end_node is not None:
            end_node = end_node.next
            nth_node = nth_node.next
        return nth_node
    def print_list(self):
        if (self.is_empty()):
            print('List is empty')
        temp = self.head
        while temp.next is not None:
            print(temp.data, end=' -> ')
            temp = temp.next
        print(temp.data, '-> None')
        return True

class Node:
    def __init__(self, val) -> None:
        self.data = val
        self.prev = None
        self.next = None
class LinkedList:
    def __init__(self) -> None:
        self.head = None
        self.tail = None
    def delete(self, val):
        deleted = False
        if (self.is_empty()):
            print('List is empty')
            return deleted
        curr = self.get_head()
        if curr.data is val:
            self.head = curr.next
            if (curr.next != None and curr.next.prev != None):
                curr.next.prev = None
                deleted = True
                print(str(curr.data) + ' deleted')
            return deleted
        while curr:
            if curr.data is val:
                if curr.next:
                    prev = curr.prev
                    next = curr.next
                    prev.next = next
                    next.prev = prev
                else:
                    curr.prev = None
                deleted = True
                break
            curr = curr.next
        if deleted is False:
            print(str(val) + ' is not in the list')
        else:
            print(str(val) + ' deleted')
        return deleted
class CircularLinkedList:
    def __init__(self) -> None:
        self.head = None
    def prepend(self, data):
        new = Node(data)
        curr = self.head
        new.next = self.head
        if not self.head:
            new.next = new
        else:
            while curr != self.head:
                curr = curr.next
            curr.next = new
        self.head = new
    def append(self, data):
        if not self.head:
            self.head = Node(data)
            self.head.next = self.head
        else:
            new = Node(data)
            curr = self.head
            while curr.next != self.head:
                curr = curr.next
            curr.next = new
            new.next = self.head
    def print_list(self):
        curr = self.head
        while curr:
            print(curr.data)
            curr = curr.next
            if curr == self.head:
                break
    def remove(self, key):
        if self.head:
            if self.head.data == key:
                curr = self.head
                while curr.next != self.head:
                    curr = curr.next
                if self.head == self.head.next:
                    self.head = None
                else:
                    curr.next = self.head.next
                    self.head = self.head.next
            else:
                curr = self.head
                prev = None
                while curr.next != self.head:
                    prev = curr
                    curr = curr.next
                if curr.data == key:
                    prev.next = curr.next
                    curr = curr.next

# Queue
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    def get_head(self):
        if (self.head != None):
            return self.head.data
        else:
            return False
    def is_empty(self):
        if (self.head is None):
            return True
        else:
            return False
    def insert_tail(self, data):
        new = Node(data)
        if(self.is_empty()):
            self.head = new
            self.tail = new
        else:
            self.tail.next = new
            new.prev = self.tail
        self.length += 1
        return new
    def remove_head(self):
        if (self.is_empty()):
            return False
        node_to_remove = self.head
        if (self.length == 1):
            self.head = None
            self.tail = None
        else:
            self.head = node_to_remove.next
            self.head.prev = None
            node_to_remove.next = None
        self.length -= 1
        return node_to_remove.data
    def tail_node(self):
        if (self.head != None):
            return self.tail.data
        else:
            return False
    def __str__(self):
        val = ''
        if (self.is_empty()):
            return ''
        temp = self.head
        val = '[' + str(temp.data) + ', '
        temp = temp.next
        while temp.next:
            val = val + str(temp.data) + ', '
            temp = temp.next
        val = val + str(temp.data) + ']'
        return val
class MyQueue:
    def __init__(self):
        self.items = DoublyLinkedList()
    def is_empty(self):
        return self.items.length == 0
    def front(self):
        if self.is_empty():
            return None
        return self.items.get_head()
    def rear(self):
        if self.is_empty():
            return None
        return self.items.tail_node()
    def size(self):
        return self.items.length
    def enqueue(self, val):
        return self.items.insert_tail(val)
    def dequeue(self, val):
        return self.items.remove_head()
    def print_list(self):
        return self.items.__str__()
# Stack
class MyStack:
    def __init__(self):
        self.list = []
        self.size = 0
    def is_empty(self):
        return self.size == 0
    def peek(self):
        if self.is_empty():
            return None
        return self.list[-1]
    def size(self):
        return self.size
    def push(self, val):
        self.size += 1
        self.list.append(val)
    def pop(self):
        if self.is_empty():
            return None
        self.size -= 1
        return self.list.pop()
# Tree
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
    def insert(self, val):
        curr = self
        parent = None
        while curr:
            parent = curr
            if val < curr.val:
                curr = curr.left
            else:
                curr = curr.right
        if (val < parent.val):
            parent.left = Node(val)
        else:
            parent.right = Node(val)
    def insert(self, val):
        if val < self.val:
            if self.left:
                self.left.insert(val)
            else:
                self.left = Node(val)
                return
        else:
            if self.right:
                self.right.insert(val)
            else:
                self.right = Node(val)
                return
    def search(self, val):
        curr = self
        while curr is not None:
            if val < curr.val:
                curr = curr.left
            elif val > curr.val:
                curr = curr.right
            else:
                return True
        return False
    def search(self, val):
        if val < self.val:
            if self.left:
                return self.left.search(val)
            else:
                return False
        elif val > self.val:
            if self.right:
                return self.right.search(val)
            else:
                return False
        else:
            return True
        return False
    def copy(self, node2): # CHECK THIS
        self.val = node2.val
        if(node2.left):
            self.left = node2.left
        if(node2.right):
            self.right = node2.right
    #Frame 
    def delete(self, val):
        if val < self.val:
            pass
        elif val > self.val:
            pass
        else:
            pass
    def delete(self, val):
        if val < self.val:
            if self.left:
                self.left = self.left.delete(val)
            else:
                print('Not found')
                return self
        elif val > self.val:
            if self.right:
                self.right = self.right.delete(val)
            else:
                print('Not found')
                return self
        else:
            if self.left is None and self.right is None:
                self = None # ???
                return None
            elif self.left is None:
                temp = self.right
                self = None
                return temp
            elif self.right is None:
                temp = self.left
                self = None
                return temp
            else: # ???
                curr = self.right
                while curr.left is not None:
                    curr = curr.left
                self.val = curr.val
                self.right = self.right.delete(curr.val)
        return self
# Traversal
def pre_order(node):
    print(node.val)
    pre_order(node.left)
    pre_order(node.right)
def post_order(node):
    post_order(node.left)    
    post_order(node.right)
    print(node.val)
def in_order(node):
    in_order(node.left)
    print(node.val)
    in_order(node.right)
class BST:
    def __init__(self, val):
        self.root = Node(val)
    def set_root(self, val):
        self.root = Node(val)
    def get_root(self):
        return self.root.get() #CHECK THIS
    def insert(self, val):
        if self.root:
            return self.root.insert(val)
        else:
            self.root = Node(val)
            return True
    def search(self, val):
        if self.root:
            return self.root.search(val)
        else:
            return False
    def delete(self, val):
        if self.root is not None:
            self.root = self.root.delete(val)
    def height(self, node):
        if node is None:
            return -1
        left_height = self.height(node.left)
        right_height = self.right(node.right)
        return 1 + max(left_height, right_height)
    def size(self, node):
        if node is None:
            return 0
        return 1 + self.size(node.left) + self.size(node.right)
    def valid_BST(self):
        def helper(node, lo = float('-inf'), up = float('inf')):
            if not node:
                return True
            val = node.data
            if val <= lo or val >= up:
                return False
            if not helper(node.right, val, up):
                return False
            if not helper(node.left, lo, val):
                return False
            return True
        return helper(self.root)
#Binary Search
def binary_search(lo, hi, cond):
    while lo <= hi:
        mid = (lo + hi) // 2
        result = cond(mid)
        if result == 'found':
            return mid
        elif result == 'left':
            hi = mid - 1
        else:
            lo = mid + 1
    return -1
def first_position(nums, target):
    def cond(mid):
        if nums[mid] == target:
            if mid > 0 and nums[mid - 1] == target:
                return 'left'
            return 'found'
        elif nums[mid] < target:
            return 'right'
        else:
            return 'left'
    return binary_search(0, len(nums) - 1, cond)
def last_position(nums, target):
    def cond(mid):
        if num[mid] == target:
            if mid < len(nums) -1 and num[mid + 1] == target:
                return 'right'
            return 'found'
        elif nums[mid] < target:
            return 'right'
        else:
            return 'left'
    return binary_search(0, len(nums) - 1, cond)
def first_last_position(nums, target):
    return first_position(nums, target), last_position(nums, target)

### *** labuladong ***



#Advanced Python



# TODO: sort and search algo