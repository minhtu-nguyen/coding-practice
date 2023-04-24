n = 0
n = 'abc'
n, m = 0, 'abc'
n += 1
n = None
n = 1
if n > 2:
  n -= 1
elif n == 2:
  n *= 2
else:
  n += 2

n, m = 1, 2
if ((n > 2 and n != m) or n == m):
  n += 1

n = 0
while n < 5:
  print(n)
  n += 1
for i in range(5):
  print(i)
for i in range(2, 6):
  print(i)
for i in range(5, 1, -1):
  print(i)

print(5 / 2)
print(5 // 2)
print(-3 // 2)
print(int(-3 / 2))
print(10 % 3)
print(-10 % 3)

import math
from multiprocessing import  heap
print(math.fmod(-10, 3))
print(math.floor(3 / 2))
print(math.ceil(3 / 2))
print(math.sqrt(2))
print(math.pow(2, 3))
print(math.pow(2, 200))

float("inf")
float("-inf")

arr = [1, 2, 3]
arr.append(4)
arr.pop()
arr.insert(1, 5)

n = 5
arr = [1] * n
print(arr[-1])
print(arr[1:3])
a, b, c = [1, 2, 3]

nums = [1, 2, 3]
for n in nums:
  print(n)
for i, n in enumerate(nums):
  print(i, n)

for n1, n2 in zip(nums, nums):
  print(n1, n2)

nums.reverse()

arr = [5, 4, 7, 3, 8]
arr.sort()
arr.sort(reverse=True)
arr.sort(key = lambda x: len(x))

arr = [i for i in range(5)]

arr = [[0] * 4 for i in range(4)]

s = "abc"
print(s[0:2])
s += 'def'

print(ord('a'))

strings = ['ab', 'cd', 'ef']
print(''.join(strings))

from collections import Counter, defaultdict, deque

queue = deque()
queue.append(1)
queue.append(2)
queue.popleft()
queue.appendleft(1)
queue.pop()

mySet = set()

mySet.add(1)
mySet.add(2)
print(1 in mySet)
mySet.remove(2)

print(set([1, 2, 3]))

mySet = {i for i in range(5)}

myMap = {}
myMap['alice'] = 88
myMap['bob'] = 77
print('alice' in myMap)
myMap.pop()

myMap = {i: 2 * i for i in range(3)}

for key in myMap:
  print(key, myMap[key])

for val in myMap.values():
  print(val)

for key, val in myMap.items():
  print(key, val)

tup = (1, 2, 3)

print(tup[0])

myMap[(1, 2)] = 3
mySet.add((1, 2))
print((1, 2) in mySet)

import heapq

minHeap = []
heapq.heappush(minHeap, 3)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 4)
print(minHeap[0])

while len(minHeap):
  print(heapq.heappop(minHeap))

maxHeap = []
heapq.heappush(maxHeap, -3)
heapq.heappush(maxHeap, -2)
heapq.heappush(maxHeap, -4)
print(-1 * maxHeap[0])

while len(maxHeap):
  print(-1 * heapq.heappop(maxHeap))

arr = [2, 1, 8, 4, 5]
heapq.heapify(arr)
while arr:
  print(heapq.heappop(arr))

def outer(a, b):
  c = 'c'

  def inner():
    return a + b + c
  return inner()

def double(arr, val):
  def helper():
    for i, n in enumerate(arr):
      arr[i] *= 2
    nonlocal val
    val *= 2
  helper()
  print(arr, val)

class MyClass:
  def __init__(self, nums):
    self.nums = nums
    self.size = len(nums)

  def getLength(self):
    return self.size

  def getDoubleLength(self):
    return 2 * self.getLength

myObj = MyClass([1, 2, 3])

arr = [3, 2, 1]
arr.append(4)
arr.insert(2, 5)
arr.remove(5)
arr.pop()
arr.reverse()
del arr[::2]

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
    if self.head is None:
      self.head = new_node
      return self.head
    temp = self.head
    while temp.next:
      temp = temp.next
    temp.next = new_node
    return self.head

  def search(self, val):
    curr = self.head
    while curr:
      if curr.data == val:
        return True
      curr = curr.next
    return False
  
  def insert_node(self, prev_node, data):
    if not prev_node:
      print('Previous does not exist')
    new_node = Node(data)
    new_node.next = prev_node.next
    prev_node.next = new_node

  def delete_at_head(self):
    head = self.head
    if self.head:
      self.head = head.next
      head = None
    return self.head
  
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
    while curr:
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
    curr = self.head
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

    while slo and fas.next:
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

  def find_nth(self, n):
    if self.is_empty():
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

class Node:
  def __init__(self, val):
    self.data = val
    self.prev = None
    self.next = None

class DoublyLinkedList:
  def __init__(self):
    self.head = None
    self.tail = None

  def delete(self, val):
    deleted = False
    if self.is_empty():
      print('List is empty')
      return deleted
    curr = self.head
    if curr.data is val:
      self.head = curr.next
      if (curr.next != None and curr.next.prev != None):
        curr.next.prev = None
        deleted = True
      return deleted
    while curr:
      if curr.data  is val:
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
    return deleted

class CircularLinkedList:
  def __init__(self):
    self.head = None

  def prepend(self, data):
    new = Node(data)
    if not self.head:
      new.next = new
      return
    curr = self.head
    new.next = self.head
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
    if self.head != None:
      return self.head.data
    else:
      return False

  def is_empty(self):
    if self.head is None:
      return True
    else:
      return False

  def insert_tail(self, data):
    new = Node(data)
    if (self.is_empty()):
      self.head = new
      self.tail = new
    else:
      self.tail.next = new
      new.prev = self.tail
    self.length += 1
  
  def remove_head(self):
    if (self.is_empty()):
      return False
    node_to_remove = self.head
    if self.length == 1:
      self.head = None
      self.tail = None
    else:
      self.head = node_to_remove.next
      self.head.prev = None
      node_to_remove.next = None

class MyQueue:
  def __init__(self):
    self.items = DoublyLinkedList()

  def front(self):
    if self.is_empty():
      return None
    return self.items.get_head()

  def enqueue(self, val):
    return self.items.insert_tail(val)
  def dequeue(self, val):
    return self.items.remove_head()

class MyStack:
  def __init__(self):
    self.list = []
    self.size = 0

  def is_empty(self):
    return self.size == 0
  
  def peek(self):
    if self.is_empty():
      return False
    return self.list[-1]
  
  def push(self, val):
    self.size += 1
    self.list.append(val)
  
  def pop(self):
    if self.is_empty():
      return None
    self.size -= 1
    return self.list.pop()

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
    if val < parent.val:
      parent.left = Node(val)
    else:
      parent.right = Node(val)

  def insert_recur(self, val):
    if val < self.val:
      if self.left:
        self.left.insert_recur(val)
      else:
        self.left = Node(val)
        return
    else:
      if self.right:
        self.right.insert_recur(val)
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

  def search_recur(self, val):
    if val < self.val:
      if self.left:
        return self.left.search_recur(val)
      else:
        return False
    elif val > self.val:
      if self.right:
        return self.right.search_recur(val)
      else:
        return False
    else:
      return True

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
        self = None
        return None
      elif self.left is None:
        temp = self.right
        self = None
        return temp
      elif self.right is None:
        temp = self.left
        self = None
        return temp
      else:
        curr = self.right
        while curr.left is not None:
          curr = curr.left
        self.val = curr.val
        self.right = self.right.delete(curr.val)

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
  
  def set_root(self,  val):
    self.root = Node(val)

  def insert(self, val):
    if self.root:
      return self.root.insert(val)
    else:
      return False
  def height(self, node):
    if node is None:
      return -1
    left_height = self.height(node.left)
    right_height = self.height(node.right)
    return 1 + max(left_height, right_height)

  def size(self, node):
    if node is None:
      return 0
    return 1 + self.size(node.left) + self.size(node.right)

  def valid_BST(self):
    def helper(node, lo = float('-inf'), hi = float('inf')):
      if not node:
        return True
      val = node.val
      if val <= lo or val >= hi:
        return False
      if not helper(node.right, val, hi):
        return False
      if not helper(node.left, lo, val):
        return False
      return True
    return helper(self.root)

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
    if nums[mid] == target:
      if mid < len(nums) - 1 and nums[mid + 1] == target:
        return 'right'
      else:
        return 'found'
    elif nums[mid] < target:
      return 'right'
    else:
      return 'left'
  return binary_search(0, len(nums) - 1, cond)

print(complex(10, 20))

my_string = "This is MY string!"
print(my_string[0:4])
print(my_string[:4])
print(my_string[0:10:2])
print(my_string[::-1])

num1 = 10
num2 = 20

print(num1 & num2)
print(num1 | num2)
print(num1 ^ num2)
print(~num1)
print(num1 << 3)
print(num2 >> 3)

random_string = "This is a string"
print(random_string.find("is"))

a_string = "Welcome to Educative!"
new_string = a_string.replace("Welcome to", "Greetings from")

print("uppercase".upper())
print("lower".lower())

print(f'{num1} formatted printing')

triple = lambda num : num * 3 
my_func = lambda num : "High" if num > 50 else "Low"

nums = [0, 1, 2, 3, 4, 5]
double_nums = map(lambda n : n * 2, nums)
greater_than_3 = list(filter(lambda n : n > 3), nums)

def rec_count(number):
  print(number)
  if number == 0:
    return 0
  rec_count(number - 1)
  print(number)

def fib(n):
  first = 0
  second = 1
  if n < 1:
    return -1
  if n == 1:
    return first
  if n == 2: 
    return second
  
  count = 3
  while count <= n:
    fib_n = first + second
    first = second
    second = fib_n
    count += 1
  return fib_n

class ClassName:
  pass

obj = ClassName()

class Employee:
  ID = None
  salary = None
  department = None

Steve = Employee()
Steve.ID = 3789
Steve.salary = 2500
Steve.department = "HR"

Steve.title = "Manager"

class Employee:
  def __init__(self, ID, salary = 0, department = None):
    self.ID = ID
    self.salary = salary
    self.department = department

class Player:
  teamName = 'Liverpool'
  teamMembers = []

  def __init__(self, name):
    self.name = name
    self.formerTeams = []
    self.teamMembers.append(self.name)

class Employee:
  def __init__(self, ID = None, salary = None, department = None):
    self.ID = ID
    self.salary = salary
    self.department = department

  def tax(self):
    return self.salary * 0.2
  
  def salaryPerDay(self):
    return self.salary / 30
  
class Player:
  teamName = 'Liverpool'

  def __init__(self, name):
    self.name = name

  @classmethod
  def getTeamName(cls):
    return cls.teamName
  
  @staticmethod
  def bmi(weight, height):
    return weight / (height ** 2)
  
class Employee:
  def __init__(self, ID, salary):
    self.ID = ID
    self.__salary = salary

  def __displayID(self):
    print(self.ID)

Steve = Employee(3759, 2500)
print(Steve._Employee__salary)

class user:
  def __init__(self, username = None):
    self.__username = username

  def setUsername(self, x):
    self.__username = x

  def getUsername(self):
    return self.__username
  
class Vehicle:
  def __init__(self, make, color, model):
    self.make = make
    self.color = color
    self.model = model

class Car(Vehicle):
  def __init__(self, make, color, model, doors):
    Vehicle.__init__(self, make, color, model)
    self.doors = doors
  
class Vehicle:
  fuelCap = 90

  def display(self):
    print("I am from the Vehicl Class")

class Car(Vehicle):
  fuelCap = 50

  def display(self):
    print("Fuel cap from the Vehicle Class: ", super().fuelCap)
    print("Fuel cap from the Car Class", self.fuelCap)

  def display2(self):
    super().display()
    print("I am from the Car Class")

  
class ParentClass():
  def __init__(self, a, b):
    self.a = a
    self.b = b

class ChildClass(ParentClass):
  def __init__(self, a, b, c):
    super().__init__(a, b)
    self.c = c

class HybidCard(Car):
  def turnOnHybrid(self):
    print("Hybrid mode is one")

class Truck(Vehicle):
  pass

class Combustion():
  pass
class Electric():
  pass
class Hybrid(Combustion, Electric):
  pass

class Engine:
  def setPower(self, power):
    self.power =  power

class Combustion(Engine):
  def setTankCapacity(self, tankCapacity):
    self.tankCapacity = tankCapacity

class Electric(Engine):
  def setChargeCapacity(self, chargeCapacity):
    self.chargeCapacity = chargeCapacity

class Hybrid(Combustion, Electric):
  def printDetails(self):
    print(self.power)
    print(self.tankCapacity)
    print(self.chargeCapacity)

class Shape:
  def __init__(self):
    self.sides = 0
  def getArea(self):
    pass

class Rectangle(Shape):
  def __init__(self, width = 0, height = 0):
    self.width = width
    self.height = height
    self.sides = 4

  def getArea(self):
    return self.width * self.height
  
class Circle(Shape):
  def __init__(self, radius = 0):
    self.radius = radius

  def getArea(self):
    return self.radius * self.radius * 3.142
  
class Com:
  def __init__(self, real = 0, imag = 0):
    self.real = real
    self.imag = imag
  
  def __add__(self, other):
    temp = Com(self.real + other.real, self.imag + other.imag)
    return temp

class Dog:
  def speak(self):
    print("Woof Woof")

class Cat:
  def speak(self):
    print("Meow Meow")

class AnimalSound:
  def sound(self, animal):
    return animal.speak()
  
sound = AnimalSound()
dog = Dog()
sound.sound(dog)

from abc import ABC, abstractmethod

class Shape(ABC):
  @abstractmethod
  def area(self):
    pass

  @abstractmethod
  def perimeter(self):
    pass

class Square(Shape):
  def __init__(self, length):
    self.length = length

  def area(self):
    return self.length * self.length
  
  def perimeter(self):
    return 4 * self.length
  
class Country:
  def __init__(self, name = None, population = 0):
    self.name = name
    self.population = population

  def printDetails(self):
    pass

class Person:
  def __init__(self, name, country):
    self.name = name
    self.country = country

c = Country("Wales", 1500)
p = Person("Joe", c)

class Engine:
  def __init__(self, capacity = 0):
    self.capacity = capacity

class Tires:
  def __init__(self, tires = 0):
    self.tires = tires

class Car:
  def __init__(self, eng, tr, color):
    self.eObj = Engine(eng)
    self.tObj = Tires(tr)
    self.color = color


####
#Two Sum
def twoSum(nums, target):
  prevMap = {}

  for i, n in enumerate(nums):
    diff = target - n
    if diff in prevMap:
      return [prevMap[diff], i]
    prevMap[n] = i

#Stock
def maxProfit(prices):
  res = 0
  lowest = prices[0]

  for price in prices:
    if price < lowest:
      lowest = price
    res = max(res, price - lowest)
  return res

#Contains Duplicate
def containsDuplicate(nums):
  hashset = set()

  for n in nums:
    if n in hashset:
      return True
    hashset.add(n)
  return False

#Product Except Self
def productExceptSelf(nums):
  res = [1] * len(nums)

  prefix = 1
  for i in range(len(nums)):
    res[i] = prefix
    prefix *= nums[i]
  postfix = 1
  for i in range(len(nums) - 1, -1, -1):
    res[i] *= postfix
    postfix *= nums[i]
  return res

#Maximum Subarray
def maxSubArray(nums):
  res = nums[0]
  total = 0

  for n in nums:
    total += n
    res = max(res, total)
    if total < 0:
      total = 0
  return res

#Maximum Product
def maxProduct(nums):
  res = nums[0]
  curMin, curMax = 1, 1

  for n in nums:
    temp = curMax * n
    curMax = max(n * curMax, n * curMin, n)
    curMin = min(temp, n * curMin, n)
    res = max(res, curMax)
  return res

#Find Minimum Rotated Sorted Array
def findMin(nums):
  start, end = 0, len(nums) - 1
  while start < end:
    mid = (start + end) // 2
    curr_min = min(curr_min, nums[mid])

    if nums[mid] > nums[end]:
      start = mid + 1
    else:
      end = mid - 1
  return min(curr_min, nums[start])

#Search Rotated Sorted Array
def search(nums, target):
  l, r= 0, len(nums) - 1
  while l <= r:
    mid = (l + r) // 2
    if target == nums[mid]:
      return mid
    if nums[l] <= nums[mid]:
      if target > nums[mid] or target < nums[l]:
        l = mid + 1
      else:
        r = mid - 1
    else:
      if target < nums[mid] or target > nums[r]:
        r = mid - 1
      else:
        l = mid + 1
  return -1

#3Sum
def threeSum(nums):
  res = 0
  nums.sort()

  for i, a in enumerate(nums):
    if a > 0:
      break
    if i > 0 and a == nums[i - 1]:
      continue

    l, r = i + 1, len(nums) - 1
    while l < r:
      threeSum = a + nums[l] + nums[r]
      if threeSum > 0:
        r -= 1
      elif threeSum < 0:
        l += 1
      else:
        res.append([a, nums[l], nums[r]])
        l += 1
        r -= 1
        while nums[l] == nums[l - 1] and l < r:
          l += 1
    return res

#Container Water
def maxArea(heights):
  l, r = 0, len(heights) - 1
  res = 0

  while l < r:
    res = max(res, min(heights[l], heights[r]) * (r - l))
    if heights[l] < heights[r]:
      l += 1
    elif heights[r] <= heights[l]:
      r -= 1
  return res

#Skipped bit

#Missing Number
def missingNumber(nums):
  res = len(nums)

  for i in range(len(nums)):
    res += i - nums[i]
  return res

#Climbing Stairs
def climbStairs(n):
  if n <= 3:
    return n
  n1, n2 = 2, 3

  for i in range(4, n + 1):
    temp = n1 + n2
    n1 = n2 
    n2 = temp
  return n2

#Coin Change
def coinChange(coins, amount):
  dp = [amount + 1] * (amount + 1)
  dp[0] = 0

  for a in range(len(amount + 1)):
    for coin in coins:
      if a - c >= 0:
        dp[a] = min(dp[a], 1 + dp[a - c])
  return dp[amount] if dp[amount] != amount + 1 else -1


#LIS
def lengthOfLIS(nums):
  LIS = [1] * len(nums)

  for i in range(len(nums) -1 , -1, -1):
    for j in range(i + 1, len(nums)):
      if nums[i] < nums[j]:
        LIS[i] = max(LIS[i], 1 + LIS[j])
  return max(LIS)

#LCS
def longestCommonSubsequence(text1, text2):
  dp = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]

  for i in range(len(text1) - 1, -1, -1):
    for j in range(len(text2) -1, -1, -1):
      if text1[i] == text2[j]:
        dp[i][j] = 1 + dp[i + 1][j + 1]
      else:
        dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
  return dp[0][0]

#Word Break
def wordBreak(s, wordDict):
  dp = [False] * (len(s) + 1)
  dp[len(s)] = True

  for i in range(len(s) - 1, -1, -1):
    for w in wordDict:
      if (i + len(w)) <= len(s) and s[i : i + len(w)] == w:
        dp[i] = dp[i + len(w)]
      if dp[i]:
        break
  return dp[0]

#Combination Sum
def combinatioSum(candidates, target):
  res = []

  def dfs(i, cur, total):
    if total == target:
      res.append(cur.copy())
      return
    if i >= len(candidates) or total > target:
      return
    
    cur.append(candidates[i])
    dfs(i, cur, total + candidates[i])
    cur.pop()
    dfs(i + 1, cur, total)

  dfs(0, [], 0)
  return res

#House Robber
def rob(nums):
  rob1, rob2 = 0, 0

  for n in nums:
    temp = max(n + rob1, rob2)
    rob1 = rob2
    rob2 = temp
  return rob2

#House Robber II
def rob(nums):
  return max(nums[0], helper(nums[1:], helper[:-1]))

def helper(nums):
  rob1, rob2 = 0, 0

  for n in nums:
    newRob = max(n + rob1, rob2)
    rob1 = rob2
    rob2 = newRob
  return rob2

#Decode Ways
def numDecodings(s):
  dp = {len(s): 1}

  def dfs(s):
    if i in dp:
      return dp[i]
    if s[i] == '0':
      return 0
    
    res = dfs(i + 1)
    if i + 1 < len(s) and (s[i] == 1 or s[i] == 2 and s[i + 1] in '01234556'):
      res += dfs(i + 2)
    dp[i] = res
    return res
  
  return dfs(0)

def decodeWays(s):
  dp = {len(s) : 1}

  for i in range(len(s) - 1, -1, -1):
    if s[i] == '0':
      dp[i] = 0
    else:
      dp[i] = dp[i + 1]
    
    if i + 1 < len(s) and (s[i] == '1' or (s[i] == '2' and s[i + 1] in '0123456')):
      dp += dp[i + 2]
  return dp[0]

#Unique Paths
def uniquePaths(m, n):
  row = [1] * n

  for i in range(m - 1):
    newRow = [1] * n
    for j in range(n-2, -1, -1):
      newRow[j] = newRow[j + 1] + row[j]
    row = newRow
    return row[0]
  
#Jump Game
def canJump(nums):
  goal = len(nums) - 1

  for i in range(len(nums) - 2, -1, -1):
    if i + nums[i] >= goal:
      goal = i
  return goal == 0

#Clone Graph
def cloneGraph(node):
  oldToNew = {}

  def dfs(node):
    if node in oldToNew:
      return oldToNew[node]
    copy = Node(node.val)
    oldToNew[node] = copy
    for nei in node.neighbors:
      copy.neighbors.append(dfs(nei))
    return copy
  return dfs(node) if node else None

#Course Schedule
def canFinish(numCourses, prerequesites):
  preMap = {i: [] for i in range(numCourses)}

  for crs, pre in prerequesites:
    preMap[crs].append(pre)

  visiting = set()

  def dfs(crs):
    if crs in visiting:
      return False
    if preMap[crs] == []:
      return True
    
    visiting.add(crs)
    for pre in preMap[crs]:
      if not dfs(pre):
        return False
    visiting.remove(crs)
    preMap[crs] = []
    return True
  
  for c in range(numCourses):
    if not dfs(c):
      return False
  
  return True

#Pacific Atlantic
def pacificAtlantic(heights):
  ROWS, COLS = len(heights), len(heights)
  pac, atl = set(), set()

  def dfs(r, c, visit, prevHeight):
    if (
      (r, c) in visit
      or r < 0
      or c < 0
      or r == ROWS
      or c == COLS
      or heights[r][c] < prevHeight
    ):
      return 
    
    visit.add((r, c))
    dfs(r + 1, c, visit, heights[r][c])
    dfs(r - 1, c, visit, heights[r][c])
    dfs(r, c + 1, visit, heights[r][c])
    dfs(r, c - 1, visit, heights[r][c])

  for c in range(COLS):
    dfs(0, c, pac, heights[0][c])
    dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])

  for r in range(ROWS):
    dfs(r, 0, pac, heights[r][0])
    dfs(r, COLS - 1, atl, heights[r][COLS -1])

  res = []
  for r in range(ROWS):
    for c in range(COLS):
      if (r, c) in pac and (r, c) in atl:
        res.append([r, c])
  return res

#Islands
def numIsland(grid):
  if not grid or not grid[0]:
    return 0

  islands = 0
  visit = set()
  rows, cols = len(grid), len(grid[0])

  def dfs(r, c):
    if (
      r not in range(rows)
      or c not in range(cols)
      or grid[r][c] == '0'
      or (r, c) in visit  
    ):
      return 
    
    visit.add(r, c)
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    for dr, dc in directions:
      dfs(r + dr, c + dc)

  def bfs(r, c):
    q = deque()
    visit.add((r, c))
    q.append((r, c))

    while q:
      row, col = q.popleft()
      directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

      for dr, dc in directions:
        r, c = row + dr, col + dc
        if (r) in range(rows) and c in range(cols) and grid[r][c] == '1' and (r, c) not in visit:
          q.append((r, c))
          visit.add((r, c))

  for r in range(rows):
    for c in range(cols):
      if grid[r][c] == '1' and (r, c) not in visit:
        islands += 1
        dfs(r, c)
  
  return islands

#LCS
def longestConsecutive(nums):
  numSet = set(nums)
  longest = 0

  for n in nums:
    if (n - 1) not in numSet:
      length = 1
      while (n + length) in numSet:
        length += 1
      longest = max(length, longest)
    return longest

#Alien
def alienROder(words):
  adj = {char: set() for word in words for char in word}

  for i in range(len(words) -1):
    w1, w2 = words[i], words[i + 1]
    minLen = min(len(w1), len(w2))
    if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
      return ""
    for j in range(minLen):
      if w1[j] != w2[j]:
        print(w1[j], w2[j])
        adj[w1[j].add(w2[j])]
        break
  
  visited = {}
  res = {}

  def dfs(char):
    if char in visited:
      return visited[char]
    
    visited[char] = True

    for neighChar in adj[char]:
      if dfs(neighChar):
        return True
    
    visited[char] = False
    res.append(char)

  for char in adj:
    if dfs(char):
      return ""
    
    res.reverse()
    return "".join(res)

#Graph Valid Tree
def validTree(n, edges):
  if not n:
    return True
  adj = {i: [] for i in range(n)}
  for n1, n2 in edges:
    adj[n1].append(n2)
    adj[n2].append(n1)

  visit = set()
  def dfs(i, prev):
    if i in visit:
      return False
    visit.add(i)

    for j in adj[i]:
      if j == prev:
        continue
      if not dfs(j, i):
        return False
    return True
  
  return dfs(0, -1) and n == len(visit)

#Insert Interval
def insert(intervals, newInterval):
  res = []

  for i in range(len(intervals)):
    if newInterval[1] < intervals[i][0]:
      res.append(newInterval)
      return res + intervals[i:]
    elif newInterval[0] > intervals[i][1]:
      res.append(intervals[i])
    else:
      newInterval = [min(newInterval[0], intervals[i][0]),
                     max(newInterval[1], intervals[i][1])]
  res.append(newInterval)

#merge Interval
def merge(intervals):
  intervals.sort(key = lambda pair: pair[0])
  output = [intervals[0]]

  for start, end in intervals:
    lastEnd = output[-1][1]
    if start <= lastEnd:
      output[-1][1] = max(lastEnd, end)
    else:
      output.append([start, end])
  return output

#Non-overlapping
def eraseOverlappingIntervals(intervals):
  intervals.sort()
  res = 0

  prevEnd = intervals[0][1]
  for start, end in intervals[1:]:
    if start >= prevEnd:
      prevEnd = end
    else:
      res += 1
      prevEnd = min(end, prevEnd)
  return res

#Meeting Room
def canAttendMeeting(intervals):
  intervals.sort(key = lambda i: i[0])

  for i in range(1, len(intervals)):
    i1 = intervals[i - 1]
    i2 = intervals[i]

    if i1[1] > i2[0]:
      return False
  return True

#Meeting II
def minMeetingRooms(intervals):
  time = []
  for start, end in intervals:
    time.append((start, 1))
    time.append((end, -1))
  time.sort(key = lambda x: (x[0], x[1]))

  count = 0
  max_count = 0
  for t in time:
    count += t[1]
    max_count = max(max_count, count)
  return max_count

#Reverse LL
def reverseList(head):
  prev, curr = None, head

  while curr:
    temp = curr.next
    curr.next = prev
    prev = curr
    curr = temp
  return prev

#Detect Cycle
def hasCycle(head):
  slow, fast = head, head

  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
      return True
  return False

#Merge Sorted List
def mergeTwoLists(list1, list2):
  dummy = LinkedList()
  tail = dummy

  while list1 and list2:
    if list1.val < list2.val:
      tail.next = list1
      list1 = list1.next
    else:
      tail.next = list2
      list2 = list2.next

  if list1:
    tail.next = list1
  elif list2:
    tail.next = list2
  return dummy

#Merge K Sorted Lists:
def mergeKLists(lists):
  if not lists or len(lists) == 0:
    return None
  
  while len(lists) > 1:
    mergedLists = []
    for i in range(0, len(lists), 2):
      l1 = lists[i]
      l2 = lists[i + 1] if (i + 1) < len(lists) else None
      mergedLists.append(mergeList(l1, l2))
    lists = mergedLists
  return lists[0]

def mergeList(l1, l2):
  dummy = LinkedList()
  tail = dummy

  while l1 and l2:
    if l1 < l2:
      tail.next = l1
      l1 = l1.next
    else:
      tail.next = l2
      l2 = l2.next

  if l1:
    tail.next = l1
  if l2:
    tail.next = l2
  return dummy.next

#Remove nth node
def removeNthFromEnd(head, n):
  dummy = LinkedList()
  left = dummy
  right = dummy

  while n > 0:
    right = right.next
    n -= 1
  while right:
    left = left.next
    right = right.next
  
  left.next = left.next.next
  return dummy.next

#Reorder Lists
def reorderList(head):
  slow, fast = head, head.next
  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

  second = slow.next
  prev = slow.next = None
  while second:
    tmp = second.next 
    second.next = prev
    prev == second
    second = tmp

  first, second = head, prev
  while second:
    tmp1, tmp2 = first.next, second.next
    first.next = second
    second.next = tmp1
    first, second = tmp1, tmp2

#Set Matrix Zeroes
def setZeroes(matrix):
  ROWS, COLS = len(matrix), len(matrix[0])

  for r in range(ROWS):
    for c in range(COLS):
      if matrix[r][c] == 0:
        matrix[0][c] = 0
        if r > 0:
          matrix[r][0] = 0
        else:
          rowZero = True
  
  for r in range(1, ROWS):
    for c in range(1, COLS):
      if matrix[0][c] == 0 or matrix[r][0] == 0:
        matrix[r][c] = 0

  if matrix[0][0] == 0:
    for r in range(ROWS):
      matrix[r][0] = 0

  if rowZero:
    for c in range(COLS):
      matrix[0][c] = 0

#Spiral
def spiralOrder(matrix):
  res = []
  left, right = 0, len(matrix[0])
  top, bottom = 0, len(matrix)

  while left < right and top < bottom:
    for i in range(left, right):
      res.append(matrix[top][i])
    top += 1
    for i in range(top, bottom):
      res.append(matrix[i][right - 1])
    right -= 1
    if not (left < right and top < bottom):
      break
    for i in range(right - 1, left -1, -1):
      res.append(matrix[bottom - 1][i])
    bottom -= 1
    for i in range(bottom - 1, top - 1, -1):
      res.append(matrix[i][left])
    left += 1
  return res

#Rotate Image
def rotate(matrix):
  l, r = 0, len(matrix) - 1
  while l < r:
    for i in range(r - l):
      top, bottom = l, r

      topLeft = matrix[top][l + i]
      matrix[top][l + i] = matrix[bottom - i][l]
      matrix[bottom - i][l] = matrix[bottom][r - i]
      matrix[bottom][r - i] = matrix[top + i][r]
      matrix[top + i][r] = topLeft
    r -= 1
    l += 1

#Word Search
def exist(board, word):
  ROWS, COLS = len(board), len(board[0])
  path = set()

  def dfs(r, c, i):
    if i == len(word):
      return True
    if (
      min(r, c) < 0
      or r >= ROWS
      or c >= COLS
      or word[i] != board[r][c]
      or (r, c) in path
    ):
      return False
    
    path.add((r, c))
    res = (
      dfs(r + 1, c, i + 1)
      or dfs(r - 1, c, i + 1)
      or dfs(r, c + 1, i = 1)
      or dfs(r, c - 1, i + 1)
    )
    path.remove((r, c))
    return res
  
  count = defaultdict(int, sum(map(Counter, board), Counter()))
  if count[word[0]] > count[word[-1]]:
    word = word[::-1]

  for r in range(ROWS):
    for c in range(COLS):
      if dfs(r, c, 0):
        return True
  return False

#Longest Substring Without Repeating
def lengthOfLongestSubstring(s):
  charSet = ()
  l = 0
  res = 0

  for r in range(len(s)):
    while s[r] in charSet:
      charSet.remove(s[l])
      l += 1
    charSet.add(s[r])
  return res

#Character Replacement
def characterReplacement(s, k):
  count = {}
  res = 0

  l = 0
  maxf = 0
  for r in range(len(s)):
    count[s[r]] = 1 + count.get(s[r], 0)
    maxf = max(maxf, count[s[r]])

    if (r - l + 1) - maxf > k:
      count[s[l]] -= 1
      l += 1
    res = max(res, r - l + 1)
  return res

#Minimum Substring
def minWindow(s, t):
  if t == "":
    return ""
  countT, window = {}, {}
  for c in t:
    countT[c] = 1 + countT.get(c, 0)
  have, need = 0, len(countT)
  res, resLen = [-1, -1]. float("-inf")
  l = 0

  for r in range(len(s)):
    c = s[r]
    window[c] = 1 + window.get(c, 0)

    if c in countT and window[c] == countT[c]:
      have += 1
    while have != need:
      if (r - l + 1) < resLen:
        res = [l, r]
        resLen = r - l + 1
      window[s[l]] -= 1
      if s[l] in countT and window[s[l]] < countT[s[l]]:
        have -= 1
      l += 1
  l, r = res
  return s[l : r + 1] if resLen != float('inf') else ""

#Valid Anagram
def isAnagram(s, t):
  if len(s) != len(t):
    return False
  countS, countT = {}, {}

  for i in range(len(s)):
    countS[s[i]] = 1 + countS.get(s[i], 0)
    countT[t[i]] = 1 + countT.get(t[i], 0)
  return countS == countT

#Group Anagram
def groupAnagram(strs):
  ans = defaultdict()
  for s in strs:
    count = [0] * 26
    for c in s:
      count[ord(c) - ord('a')] += 1
    ans[tuple(count)].append(s)
  return ans.values()

#Valid Parenthesis
def validParenthesis(s):
  Map = {')': '(', ']': '[', '}': '{'}
  stack = []

  for c in s:
    if c not in Map:
      stack.append(c)
      continue
    if not stack or stack [-1] != Map[c]:
      return False
    stack.pop()
  return not stack

#Valid Palindrome
def alphanum(c):
  return (
    ord("A") <= ord(c) <= ord("Z")
    or ord("a") <= ord(c) <= ord("z")
    or ord("0") <= ord(c) <= ord("9")
  )

def isPalindrome(s: str):
  l, r = 0, len(s) - 1
  while l < r:
    while l < r and not alphanum(s[l]):
      l += 1
    while l < r and not alphanum(s[r]):
      r -= 1
    if s[l].lower() != s[r].lower():
      return False
    l += 1
    s -= 1
  return True

#Longest Palindromic
def longestPalindrome(s):
  res = ""
  resLen = 0

  for i in range(len(s)):
    l, r = i, i 
    while l >= 0 and r < len(s) and s[l] == s[r]:
      if (r - l + 1) > resLen:
        res = s[l: r + 1]
        resLen = r - l + 1
      l -= 1
      r += 1

    l, r = i, i + 1
    while l >= 0 and r < len(s) and s[l] == s[r]:
      if (r - l + 1) > resLen:
        res = s[l: r + 1]
        resLen = r - l + 1
      l -= 1
      r += 1

  return res

#Palindromic Substring
def countPali(s, l, r):
  res = 0
  while l >= 0 and r < len(s) and s[l] == s[r]:
    res += 1
    l -= 1
    r += 1
  return res

def countSubstrings(s):
  res = 0

  for i in range(len(s)):
    res += countPali(s, i, i)
    res += countPali(s, i, i + 1)
  return res

#Encode and Decode
def encode(strs):
  res = ""
  for s in strs:
    res += str(len(strs)) + "#" + s
  return s

def decode(s):
  res, i = [], 0
  while i < len(s):
    j = i
    while s[j] != '#':
      j += 1
    length = int(s[i:j])
    res.append(s[j + 1 : j + 1 + length])
    i = j + 1 + length
  return res

#Max Depth BT
def maxDepth(root):
  if not root:
    return 0
  
  return 1 + max(maxDepth(root.left), maxDepth(root.right))

#Same Tree
def isSameTree(p, q):
  if not p and q:
    return True
  if p and q and p.val == p.val:
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
  else:
    return False
  
#Flip Tree
def invertTree(root):
  if not root:
    return None
  
  tmp = root.left
  root.left = root.right
  root.rigth = tmp

  invertTree(root.left)
  invertTree(root.right)
  return root

#BT Max Path Sum
def maxPathSum(root):
  res = [root.val]

  def dfs(root):
    if not root:
      return 0
    
    leftMax = dfs(root.left)
    rightMax = dfs(root.right)
    leftMax = max(leftMax, 0)
    rightMax = max(rightMax, 0)

    res[0] = max(res[0], root.val + leftMax + rightMax)
    return root.val + max(leftMax, rightMax)
  dfs(root)
  return res[0]

#Level Order
def levelOrder(root):
  res = []
  q = deque()
  if root:
    q.append(root)

  while q:
    val = []
    for i in range(len(q)):
      node = q.popleft()
      val.append(node.val)
      if node.left:
        q.append(node.left)
      if node.right:
        q.append(node.right)
    res.append(val)
  return res

#Serialize Deserialize

#Subtree
def sameTree(s, t):
  if not s and not t:
    return True
  if s and t and s.val == t.val:
    return sameTree(s.left, t.left) and sameTree(s.right, t.right)
  return False

def isSubtree(s, t):
  if not t:
    return True
  if not s:
    return False
  
  if sameTree(s, t):
    return True
  return isSubtree(s.left, t) or isSubtree(s.right, t)

#Construct BT
def buildTree(preorder, inorder):
  if not preorder or not inorder:
    return None

  root = Node(preorder[0])
  mid = inorder.index(root)
  root.left = buildTree(preorder[1:mid + 1], inorder[:mid])
  root.right = buildTree(preorder[mid + 1:], inorder[mid + 1:])
  return root

#Validate
def isValidBST(root):
  def valid(node, left, right):
    if not node:
      return True
    if not(node.val > left and node < right):
      return False
    
    return valid(node.left, left, node.val) and valid(node.right, node.val, right)
  return valid(root, float('-inf'), float('inf'))

#Kth Smallest
def kthSmallest(root, k):
  stack = []
  curr = root

  while stack or curr:
    while curr:
      stack.append(curr)
      curr = curr.left
    curr = stack.pop()
    k -= 1
    if k == 0:
      return curr.val
    curr = curr.right

#Lowest Common Ancestor
def lowestCommonAncestor(p, q):
  cur = root
  while cur:
    if p.val > cur.val and q.val > cur.val:
      cur = cur.right
    elif p.val < cur.val and q.val < cur.val:
      cur = cur.left
    else:
      return cur

#Trie
class TrieNode: 
  def __init__(self):
    self.children = [None] * 26
    self.end = False

class Trie:
  def __init__(self):
    self.root = TrieNode()

  def insert(self, word):
    curr = self.root
    for c in word:
      i = ord(c) - ord('a')
      if curr.children[i] == None:
        curr.children[i] = TrieNode()
      curr = curr.children[i]
    curr.end = True

  def search(self, word):
    curr = self.root
    for c in word:
      i = ord(c) - ord('a')
      if curr.children[i] == None:
        return False
      curr = curr.children[i]
    return curr.end
  
  def startsWith(self, prefix):
    curr = self.root
    for c in prefix:
      i = ord(c) - ord('a')
      if curr.children[i] == None:
        return False
      curr = curr.children[i]
    return True
  
#And and Search
class TrieNode:
  def __init__(self):
    self.children = {}
    self.word = False

class WordDict:
  def __init__(self):
    self.root = TrieNode()

  def addWord(self, word):
    cur = self.root
    for c in word:
      if c not in cur.children:
        cur.children[c] = TrieNode()
      cur = cur.children[c]
    cur.word = True

  def search(self, word):
    def dfs(j, root):
      cur = root

      for i in range(j, len(word)):
        c = word[j]
        if c = '.':
          for child in cur.children.values():
            if dfs(i + 1, child):
              return True
        else:
          if c not in cur.children:
            return False
          cur = cur.children[c]
    return dfs(0, self.root)
  

#Word Search II

#Top K
def topKFrequent(nums, k):
  count = {}
  freq = [[] for i in range(len(nums) + 1)]

  for n in nums:
    count[n] = 1 + count.get(n, 0)
  for n, c in count.items:
    freq[c].append(n)

  res = []
  for i in range(len(freq) - 1, 0, -1):
    for n in freq[i]:
      res.append(n)
      if len(res) == k:
        return res
      
#Median
class MedianFinder:
  def __init__(self):
    self.small, self.large = [], []

  def addNum(self, num):
    if self.large and num > self.large[0]:
      heapq.heappush(self.large, num)
    else:
      heapq.heappush(self.small, -1 * num)
    
    if len(self.small) > len(self.large) + 1:
      val = -1 * heapq.heappop(self.small)
      heapq.heappush(self.large, val)
    if len(self.large) > len(self.small) + 1:
      val = heapq.heappop(self.large)
      heapq.heappush(self.small, -1 * val)

  def findMedian(self):
    if len(self.small) > len(self.large):
      return -1 * self.small[0]
    elif len(self.large) > len(self.small):
      return self.large[0]
    else:
      return (-1 * self.small[0] + self.large[0]) / 2
