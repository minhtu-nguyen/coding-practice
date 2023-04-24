# Comment
'''
Comment
'''
##Variables
#Number
num = 123456789  # Assigning an integer to a variable
print(-85.6701)  # A negative float
#complex(real, imaginary)
print(complex(10, 20))  # Represents the complex number (10 + 20j)
#Boolean
print(True)
f_bool = False
#String
print("Harry Potter!")  # Double quotation marks
got = 'Game of Thrones...'  # Single quotation marks
my_string = "This is MY string!"
print(my_string[0:4]) # From the start till before the 4th index
print(my_string[1:7])
print(my_string[8:len(my_string)]) # From the 8th index till the end
print(my_string[:4])
print(my_string[8:])
#Step
print(my_string[0:7])  # A step of 1
print(my_string[0:10:2])  # A step of 2
print(my_string[13:2:-1]) # Take 1 step back each time
print(my_string[17:0:-2]) # Take 2 steps back. The opposite of what happens in the slide above
print(my_string[:8])  # All the characters before 'M'
print(my_string[8:])  # All the characters starting from 'M'
print(my_string[:])  # The whole string
print(my_string[::-1])  # The whole string in reverse (step is -1)
##Operator
print(10 + 5)
print(10 - 5)
print(40 * 10)
print(40 / 10) #A division operation always results in a floating-point number
print(43 // 10) #floored to the nearest smaller integer
print(10 % 2)
#Comparison
num1 = 5
num2 = 10
num3 = 10
list1 = [6,7,8]
list2 = [6,7,8]

print(num2 > num1)  # 10 is greater than 5
print(num1 > num2)  # 5 is not greater than 10

print(num2 == num3)  # Both have the same value
print(num3 != num1)  # Both have different values

print(3 + 10 == 5 + 5)  # Both are not equal
print(3 <= 2)  # 3 is not less than or equal to 2

print(num2 is not num3)  # Both have the same object
print(list1 is list2)  # Both have the different objects
print(list1==list2)
#Assign
num = 10
print(num)

num += 5
print(num)

num -= 5
print(num)

num *= 2
print(num)

num /= 2
print(num)

num **= 2
print(num)
#Logical
# OR Expression
my_bool = True or False
print(my_bool)

# AND Expression
my_bool = True and False
print(my_bool)

# NOT expression
my_bool = False
print(not my_bool)

##Bit
num1 = 10  # Binary value = 01010
num2 = 20  # Binary Value = 10100

print(num1 & num2)   # 0   -> Binary value = 00000
print(num1 | num2)   # 30  -> Binary value = 11110
print(num1 ^ num2)   # 30  -> Binary value = 11110
print(~num1)         # -11 -> Binary value = -(1011)
print(num1 << 3)     # 80  -> Binary value = 0101 0000
print(num2 >> 3)     # 2   -> Binary value = 0010

##String operators
print('a' < 'b')  # 'a' has a smaller Unicode value

house = "Gryffindor"
house_copy = "Gryffindor"

print(house == house_copy)

new_house = "Slytherin"

print(house == new_house)

print(new_house <= house)

print(new_house >= house)

first_half = "Bat"
second_half = "man"

full_name = first_half + second_half

random_string = "This is a random string"

print('of' in random_string)  # Check whether 'of' exists in randomString
print('random' in random_string)  # 'random' exists!

##Conditionals
if condtional statement is True:
    # execute expression1
    pass
else:
    # execute expression2
    pass

num = 10

if num > 5:
    print("The number is greater than 5")

elif num % 2 == 0:
    print("The number is even")

else:
    print("The number is odd and less than or equal to 5")

    
output_value1 if condition else output_value2

##Functions
def my_print_function():  # No parameters
    print("This")
    print("is")
    print("A")
    print("function")
# Function ended


# Calling the function in the program multiple times
my_print_function()
my_print_function()
##Built-in string functions
#Search
# a_string.find(substring, start, end)
random_string = "This is a string"
print(random_string.find("is"))  # First instance of 'is' occurs at index 2
#Replace
# a_string.replace(substring_to_be_replaced, new_string)
a_string = "Welcome to Educative!"
new_string = a_string.replace("Welcome to", "Greetings from")
print(a_string)
print(new_string)
#Letter case
print("UpperCase".upper())
print("LowerCase".lower())
#Join strings
llist = ['a', 'b', 'c']
print('>>'.join(llist)) # joining strings with >>
print('<<'.join(llist)) # joining strings with <<
print(', '.join(llist)) # joining strings with comma and space
#Format
string1 = "Learn Python {version} at {cname}".format(version = 3, cname = "Educative")
string2 = "Learn Python {0} at {1}".format(3, "Educative")
string3 = "Learn Python {} at {}".format(3, "Educative")

print(string1)
print(string2)
print(string3)

##Lambda 
triple = lambda num : num * 3  # Assigning the lambda to a variable
my_func = lambda num: "High" if num > 50 else "Low"

#Functions as arguments
def add(n1, n2):
    return n1 + n2


def subtract(n1, n2):
    return n1 - n2


def multiply(n1, n2):
    return n1 * n2


def divide(n1, n2):
    return n1 / n2


def calculator(operation, n1, n2):
    return operation(n1, n2)  # Using the 'operation' argument as a function


result = calculator(multiply, 10, 20)
print(result)
print(calculator(add, 10, 20))
#Map
num_list = [0, 1, 2, 3, 4, 5]

double_list = map(lambda n: n * 2, num_list)

print(list(double_list))
#Filter
numList = [30, 2, -15, 17, 9, 100]

greater_than_10 = list(filter(lambda n: n > 10, numList))
print(greater_than_10)

#Recursion
def rec_count(number):
    print(number)
    # Base case
    if number == 0:
        return 0
    rec_count(number - 1)  # A recursive call with a different argument
    print(number)


rec_count(5)

#Loop
for i in range(1, 11, 3):  # A sequence from 1 to 10 with a step of 3
    print(i)

float_list = [2.5, 16.42, 10.77, 8.3, 34.21]
for i in float_list:
    print(i)

for num in num_list:
    pass # You can write code here later on | break | continue

#Fibonaci iterative
def fib(n):
    # The first and second values will always be fixed
    first = 0
    second = 1

    if n < 1:
        return -1

    if n == 1:
        return first

    if n == 2:
        return second

    count = 3  # Starting from 3 because we already know the first two values
    while count <= n:
        fib_n = first + second
        first = second
        second = fib_n
        count += 1  # Increment count in each iteration
    return fib_n


n = 7
print(fib(n))

### OOP
##Class
class ClassName:
    pass

obj = ClassName()  # creating a MyClass Object
# Properties
class Employee:
    # defining the properties and assigning them None
    ID = None
    salary = None
    department = None


# cerating an object of the Employee class
Steve = Employee()

# assigning values to properties of Steve - an object of the Employee class
Steve.ID = 3789
Steve.salary = 2500
Steve.department = "Human Resources"

# creating a new attribute for Steve
Steve.title = "Manager"

#Initialize
class Employee:
    # defining the properties and assigning them None
    def __init__(self, ID, salary, department):
        self.ID = ID
        self.salary = salary
        self.department = department
#Optional/ default property
class Employee:
    # defining the properties and assigning None to them
    def __init__(self, ID=None, salary=0, department=None):
        self.ID = ID
        self.salary = salary
        self.department = department
#Defining class variables and instance variables
class Player:
    teamName = 'Liverpool'  # class variables

    def __init__(self, name):
        self.name = name  # creating instance variables

class Player:
    teamName = 'Liverpool'      # class variables
    teamMembers = []

    def __init__(self, name):
        self.name = name        # creating instance variables
        self.formerTeams = []
        self.teamMembers.append(self.name)


p1 = Player('Mark')
p2 = Player('Steve')
#Methods
class Employee:
    # defining the initializer
    def __init__(self, ID=None, salary=None, department=None):
        self.ID = ID
        self.salary = salary
        self.department = department

    def tax(self):
        return (self.salary * 0.2)

    def salaryPerDay(self):
        return (self.salary / 30)
    
    # method overloading
    def demo(self, a, b, c, d=5, e=None):
        print("a =", a)
        print("b =", b)
        print("c =", c)
        print("d =", d)
        print("e =", e)

#Class Methods and Static Methods
class Player:
    teamName = 'Liverpool'  # class variables

    def __init__(self, name):
        self.name = name  # creating instance variables

    @classmethod
    def getTeamName(cls):
        return cls.teamName


print(Player.getTeamName())

#Static 
class BodyInfo:

    @staticmethod
    def bmi(weight, height):
        return weight / (height**2)


weight = 75
height = 1.8
print(BodyInfo.bmi(weight, height))

#Access modifier
class Employee:
    def __init__(self, ID, salary):
        self.ID = ID # all properties are public
        self.__salary = salary  # salary is a private property
    def __displayID(self):  # displayID is a private method
      print("ID:", self.ID)


Steve = Employee(3789, 2500)
print("ID:", Steve.ID)  
print("Salary:", Steve.__salary)  # this will cause an error
Steve.__displayID()  # this will generate an error
print(Steve._Employee__salary)  # accessing a private property

#Encapsulation Getter/ Setter
class User:
    def __init__(self, username=None):  # defining initializer
        self.__username = username

    def setUsername(self, x):
        self.__username = x

    def getUsername(self):
        return (self.__username)


Steve = User('steve1')
print('Before setting:', Steve.getUsername())
Steve.setUsername('steve2')
print('After setting:', Steve.getUsername())

#Inheritance
class Vehicle:
    def __init__(self, make, color, model):
        self.make = make
        self.color = color
        self.model = model

    def printDetails(self):
        print("Manufacturer:", self.make)
        print("Color:", self.color)
        print("Model:", self.model)


class Car(Vehicle):
    def __init__(self, make, color, model, doors):
        # calling the constructor from parent class
        Vehicle.__init__(self, make, color, model)
        self.doors = doors

    def printCarDetails(self):
        self.printDetails()
        print("Doors:", self.doors)

#Super
class Vehicle:  # defining the parent class
    fuelCap = 90

    def display(self):  # defining display method in the parent class
        print("I am from the Vehicle Class")


class Car(Vehicle):  # defining the child class
    fuelCap = 50

    def display(self):
        # accessing fuelCap from the Vehicle class using super()
        print("Fuel cap from the Vehicle Class:", super().fuelCap)

        # accessing fuelCap from the Car class using self
        print("Fuel cap from the Car Class:", self.fuelCap)

        # defining display method in the child class
    def display2(self):
        super().display()
        print("I am from the Car Class")

class ParentClass():
    def __init__(self, a, b):
        self.a = a
        self.b = b


class ChildClass(ParentClass):
    def __init__(self, a, b, c):
        super().__init__(a, b) #not include self parameter
        self.c = c

#Types of inheritance
#Above is single inheritance
class Hybrid(Car):  # child class of Car, multi-level inheritance
    def turnOnHybrid(self):
        print("Hybrid mode is now switched on.")

#Hierarchical inheritance
class Truck(Vehicle):  # another child class of Vehicle
    pass

##Multiple inheritance
class CombustionEngine():  
    def setTankCapacity(self, tankCapacity):
        self.tankCapacity = tankCapacity


class ElectricEngine():  
    def setChargeCapacity(self, chargeCapacity):
        self.chargeCapacity = chargeCapacity

# Child class inherited from CombustionEngine and ElectricEngine
class HybridEngine(CombustionEngine, ElectricEngine):
    def printDetails(self):
        print("Tank Capacity:", self.tankCapacity)
        print("Charge Capacity:", self.chargeCapacity)

##Hybrid inheritance
class Engine:  # Parent class
    def setPower(self, power):
        self.power = power


class CombustionEngine(Engine):  # Child class inherited from Engine
    def setTankCapacity(self, tankCapacity):
        self.tankCapacity = tankCapacity


class ElectricEngine(Engine):  # Child class inherited from Engine
    def setChargeCapacity(self, chargeCapacity):
        self.chargeCapacity = chargeCapacity

# Child class inherited from CombustionEngine and ElectricEngine


class HybridEngine(CombustionEngine, ElectricEngine):
    def printDetails(self):
        print("Power:", self.power)
        print("Tank Capacity:", self.tankCapacity)
        print("Charge Capacity:", self.chargeCapacity)

##Polymorphism
class Shape:
    def __init__(self):  # initializing sides of all shapes to 0
        self.sides = 0

    def getArea(self):
        pass


class Rectangle(Shape):  # derived from Shape class
    # initializer
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.sides = 4

    # method to calculate Area
    def getArea(self):
        return (self.width * self.height)
    
class Circle(Shape):  # derived from Shape class
    # initializer
    def __init__(self, radius=0):
        self.radius = radius

    # method to calculate Area
    def getArea(self): #Overridng parent class' method
        return (self.radius * self.radius * 3.142)

#Operator overloading
class Com:
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def __add__(self, other):  # overloading the `+` operator
        temp = Com(self.real + other.real, self.imag + other.imag)
        return temp

    def __sub__(self, other):  # overloading the `-` operator
        temp = Com(self.real - other.real, self.imag - other.imag)
        return temp
    
## Duck Typing Polymorphism
class Dog:
    def Speak(self):
        print("Woof woof")


class Cat:
    def Speak(self):
        print("Meow meow")


class AnimalSound:
    def Sound(self, animal):
        animal.Speak()


sound = AnimalSound()
dog = Dog()
cat = Cat()

sound.Sound(dog)
sound.Sound(cat)

## Abstract base class
from abc import ABC, abstractmethod


class Shape(ABC):  # Shape is a child class of ABC
    @abstractmethod #prevent creating an instance of Shape class, Square is OK
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass


class Square(Shape):
    def __init__(self, length):
        self.length = length

    def area(self):
        return (self.length * self.length)

    def perimeter(self):
        return (4 * self.length)


shape = Shape()
# this code will not compile since Shape has abstract methods without
# method definitions in it

###Object Relationships
##Aggregation Has-A
class Country:
    def __init__(self, name=None, population=0):
        self.name = name
        self.population = population

    def printDetails(self):
        print("Country Name:", self.name)
        print("Country Population", self.population)


class Person:
    def __init__(self, name, country):
        self.name = name
        self.country = country

    def printDetails(self):
        print("Person Name:", self.name)
        self.country.printDetails()


c = Country("Wales", 1500)
p = Person("Joe", c)
p.printDetails()
# deletes the object p, the Country object c lives on
del p
print("")
c.printDetails()

##Composition - Part-Of
class Engine:
    def __init__(self, capacity=0):
        self.capacity = capacity

    def printDetails(self):
        print("Engine Details:", self.capacity)


class Tires:
    def __init__(self, tires=0):
        self.tires = tires

    def printDetails(self):
        print("Number of tires:", self.tires)


class Doors:
    def __init__(self, doors=0):
        self.doors = doors

    def printDetails(self):
        print("Number of doors:", self.doors)

class Car:
    '''
    We have created a Car class which contains the objects of Engine, Tires, 
    and Doors classes. Car class is responsible for their lifetime, i.e., 
    when Car dies, so does tire, engine, and doors too.
    '''
    def __init__(self, eng, tr, dr, color):
        self.eObj = Engine(eng)
        self.tObj = Tires(tr)
        self.dObj = Doors(dr)
        self.color = color

    def printDetails(self):
        self.eObj.printDetails()
        self.tObj.printDetails()
        self.dObj.printDetails()
        print("Car color:", self.color)


car = Car(1600, 4, 2, "Grey")
car.printDetails()

#### ADVANCED CONCEPTS
##Command Line Arguments
##Collections Module in Python

#ChainMap
import argparse
import os

from collections import ChainMap


def main():
    app_defaults = {"username": "admin", "password": "admin"}

    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username")
    parser.add_argument("-p", "--password")
    args = parser.parse_args()
    command_line_arguments = {key: value for key, value in vars(args).items() if value}

    chain = ChainMap(command_line_arguments, os.environ, app_defaults)
    print(chain["username"])


if __name__ == "__main__":
    main()
    os.environ["username"] = "test"
    main()


# Counter
from collections import Counter

print(Counter("superfluous"))
counter = Counter("superfluous")
print(counter["u"])

print(list(counter.elements()))

print(counter.most_common(2))

counter_one = Counter("superfluous")
print(counter_one)
counter_two = Counter("super")
print(counter_one.subtract(counter_two))
print(counter_one)

#normal dict
sentence = "The red for jumped over the fence and ran to the zoo for food"
words = sentence.split(' ')

reg_dict = {}
for word in words:
    if word in reg_dict:
        reg_dict[word] += 1
    else:
        reg_dict[word] = 1

print(reg_dict)

#default dict
from collections import defaultdict

sentence = "The red for jumped over the fence and ran to the zoo for food"
words = sentence.split(' ')

d = defaultdict(int)
for word in words: #here
    d[word] += 1

print(d)

from collections import defaultdict

animal = defaultdict(lambda: "Monkey")
animal['Sam'] = 'Tiger'

print (animal['Nick'])

print (animal)

#deque
from collections import deque
import string

d = deque(string.ascii_lowercase)
for letter in d:
    letter
d.append("bork")
print(d)


d.appendleft("test")
print(d)


d.rotate(1)
print(d)


#namedtuple
from collections import namedtuple

Parts = namedtuple("Parts", "id_num desc cost amount")
auto_parts = Parts(id_num="1234", desc="Ford Engine", cost=1200.00, amount=10)
print(auto_parts.id_num)

from collections import namedtuple

Parts = {'id_num':'1234', 'desc':'Ford Engine',
     'cost':1200.00, 'amount':10}
parts = namedtuple('Parts', Parts.keys())
print (parts)


auto_parts = parts(**Parts) #keyword arguments
print (auto_parts)

#ordered dict
from collections import OrderedDict

d = {"banana": 3, "apple": 4, "pear": 1, "orange": 2}
new_d = OrderedDict(sorted(d.items()))
print(new_d)


for key in new_d:
    print(key, new_d[key])


from collections import OrderedDict

d = {"banana": 3, "apple": 4, "pear": 1, "orange": 2}
new_d = OrderedDict(sorted(d.items()))
new_d

for key in new_d:
    key, new_d[key]
for key in reversed(new_d):
    print(key, new_d[key])

#Context Manager
with open(path, 'w') as f_obj:
    f_obj.write(some_data)
#old
f_obj = open(path, 'w')
f_obj.write(some_data)
f_obj.close()

import sqlite3


class DataConn:
    """"""

    def __init__(self, db_name):
        """Constructor"""
        self.db_name = db_name

    def __enter__(self):
        """
        Open the database connection
        """
        self.conn = sqlite3.connect(self.db_name)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the connection
        """
        self.conn.close()
        if exc_val:
            raise


if __name__ == "__main__":
    db = "test.db"
    with DataConn(db) as conn:
        cursor = conn.cursor()


from contextlib import contextmanager

@contextmanager
def file_open(path):
    try:
        f_obj = open(path, 'w')
        yield f_obj
    except OSError:
        print("We had an error!")
    finally:
        print('Closing file')
        f_obj.close()

if __name__ == '__main__':
    with file_open('test.txt') as fobj:
        fobj.write('Testing context managers')

#closinglib
from contextlib import contextmanager


@contextmanager
def closing(db):
    try:
        yield db.conn()
    finally:
        db.close()

#closing class
from contextlib import closing
from urllib.request import urlopen

with closing(urlopen("http://www.google.com")) as webpage:
    for line in webpage:
        # process the line
        pass

#contextlib.suppress(*exceptions)
from contextlib import suppress

with suppress(FileNotFoundError):
    with open('fauxfile.txt') as fobj:
        for line in fobj:
            print(line)

#redirect
from contextlib import redirect_stdout

path = 'text.txt'
with open(path, 'w') as fobj:
    with redirect_stdout(fobj):
        help(redirect_stdout)

#exit stack
from contextlib import ExitStack

filenames = []
with ExitStack() as stack:
    file_objects = [stack.enter_context(open(filename))
        for filename in filenames]
    

###functools
##lru
import urllib.error
import urllib.request

from functools import lru_cache


@lru_cache(maxsize=24)
def get_webpage(module):
    """
    Gets the specified Python module web page
    """
    webpage = "https://docs.python.org/3/library/{}.html".format(module)
    try:
        with urllib.request.urlopen(webpage) as request:
            return request.read()
    except urllib.error.HTTPError:
        return None

##partial
from functools import partial

def add(x, y):
    return x + y

p_add = partial(add, 2)
print(p_add(4))

##singledispatch
from functools import singledispatch
from decimal import Decimal


@singledispatch
def add(a, b):
    raise NotImplementedError('Unsupported type')


@add.register(float)
@add.register(Decimal)
def _(a, b):
    print("First argument is of type ", type(a))
    print(a + b)

##wraps
def another_function(func):
    """
    A function that accepts another function
    """

    @wraps(func) #display correct doc of func
    def wrapper():
        """
        A wrapping function
        """
        val = "The result of %s is %s" % (func(),
                                          eval(func())
                                          )
        return val
    return wrapper


@another_function
def a_function():
    """A pretty useless function"""
    return "1+1"


###import
#regular
import os, sys, time
import sys as system
#from sth import sth
from functools import lru_cache
#relative
from . module_y import spam as ham
import sys
sys.path.append('/usercode/my_package')
import my_package
#optional
try:
    # For Python 3
    from http.client import responses
except ImportError:  # For Python 2.5-2.7
    try:
        from httplib import responses  # NOQA
    except ImportError:  # For Python 2.4
        from BaseHTTPServer import BaseHTTPRequestHandler as _BHRH
    responses = dict([(k, v[0]) for k, v in _BHRH.responses.items()])
#local
import sys  # global scope

def square_root(a):
    # This import is into the square_root functions local scope
    import math
    return math.sqrt(a)

##importlib 
#dynamic import
 # importer.py}
import importlib
import foo


def dynamic_import(module):

    return importlib.import_module(module)


if __name__ == '__main__':
    module = dynamic_import('foo')
    module.main()

    module_two = dynamic_import('bar')
    module_two.main()
#check
import importlib.util


def check_module(module_name):
    """
    Checks if module can be imported without actually
    importing it
    """

    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        print ('Module: {} not found'.format(module_name))
        return None
    else:
        print ('Module: {} can be imported!'.format(module_name))
        return module_spec


def import_module_from_spec(module_spec):
    """
    Import the module via the passed in module specification
    Returns the newly imported module
    """

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    module_spec = check_module('fake_module')
    module_spec = check_module('collections')
    if module_spec:
        module = import_module_from_spec(module_spec)
        print (dir(module))
#from source file
import importlib.util


def import_source(module_name):
    module_file_path = module_name.__file__
    module_name = module_name.__name__

    module_spec = importlib.util.spec_from_file_location(module_name,
            module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    print (dir(module))

    msg = \
        'The {module_name} module has the following methods: {methods}'
    print (msg.format(module_name=module_name, methods=dir(module)))


if __name__ == '__main__':
    import logging
    import_source(logging)

##Iterator
#list
my_list = [1, 2, 3]
for item in iter(my_list):
    print(item)
#create iterator
class MyIterator:

    def __init__(self, letters):
        """
        Constructor
        """

        self.letters = letters
        self.position = 0

    def __iter__(self):
        """
        Returns itself as an iterator
        """

        return self

    def __next__(self):
        """
        Returns the next letter in the sequence or 
        raises StopIteration
        """

        if self.position >= len(self.letters):
            raise StopIteration
        letter = self.letters[self.position]
        self.position += 1
        return letter
    
##Generator
def silly_generator():
    yield 'Python'
    yield 'Rocks'
    yield 'So do you!'


gen = silly_generator()
print (next(gen))

print (next(gen))

print (next(gen))

print (next(gen))

with open('file.txt') as fobj:
    for line in fobj:
        #process the line

##itertools
#infinite 
from itertools import count
for i in count(10):
    if i > 20: 
        break
    else:
        print(i)

from itertools import cycle
count = 0
for item in cycle('XYZ'):
    if count > 7:
        break
    print(item)
    count += 1

from itertools import repeat
repeat(5, 5)
repeat(5, 5)

iterator = repeat(5, 5)
print (next(iterator))
print (next(iterator))
print (next(iterator))

#finite
from itertools import accumulate
import operator
print (list(accumulate(range(1, 5), operator.mul)))

from itertools import chain
my_list = ['foo', 'bar']
numbers = list(range(5))
cmd = ['ls', '/some/dir']
my_list = list(chain(['foo', 'bar'], cmd, numbers))

print (my_list)
#['foo', 'bar', 'ls', '/some/dir', 0, 1, 2, 3, 4]

from itertools import chain
numbers = list(range(5))
cmd = ['ls', '/some/dir']

print (list(chain.from_iterable([cmd, numbers])))
#['ls', '/some/dir', 0, 1, 2, 3, 4]

#compress
from itertools import compress
letters = 'ABCDEFG'
bools = [True, False, True, True, False]
print (list(compress(letters, bools)))
#['A', 'C', 'D']

#dropwhile
from itertools import dropwhile
print (list(dropwhile(lambda x: x<5, [1,4,6,4,1])))
#[6, 4, 1]

#filterfalse
from itertools import filterfalse
def greater_than_five(x):
    return x > 5 

print (list(filterfalse(greater_than_five, [6, 7, 8, 9, 1, 2, 3, 10])))
#[1, 2, 3]

#groupby
from itertools import groupby

vehicles = [('Ford', 'Taurus'), ('Dodge', 'Durango'),
            ('Chevrolet', 'Cobalt'), ('Ford', 'F150'),
            ('Dodge', 'Charger'), ('Ford', 'GT')]

sorted_vehicles = sorted(vehicles)

for key, group in groupby(vehicles, lambda make: make[0]):
    for make, model in group:
        print('{model} is made by {make}'.format(model=model,
                                                 make=make))
    print ("**** END OF GROUP ***\n")

#islice
from itertools import islice
iterator = islice('123456', 4)
print (next(iterator))
#'1'

print (next(iterator))

from itertools import islice
from itertools import count
for i in islice(count(), 3, 15):
    print(i)

from itertools import starmap
def add(a, b):
    return a+b

for item in starmap(add, [(2,3), (4,5)]):
    print(item)

#5
#9

#takewhile
from itertools import takewhile
print (list(takewhile(lambda x: x<5, [1,4,6,4,1])))
#[1, 4]

#tee
from itertools import tee
data = 'ABCDE'
iter1, iter2 = tee(data)
for item in iter1:
    print(item)

for item in iter2:
    print(item)

#zip_longest
from itertools import zip_longest
for item in zip_longest('ABCD', 'xy', fillvalue='BLANK'):
    print (item)

#('A', 'x')
#('B', 'y')
#('C', 'BLANK')
#('D', 'BLANK')

#combinations
from itertools import combinations
for item in combinations('WXYZ', 2):
    print(''.join(item))

from itertools import combinations_with_replacement
for item in combinations_with_replacement('WXYZ', 2):
    print(''.join(item))

#cartesian product
from itertools import product
arrays = [(-1,1), (-3,3), (-5,5)]
cp = list(product(*arrays))
print (cp)

#permutations
from itertools import permutations
for item in permutations('WXYZ', 2):
    print(''.join(item))

#RegEx
import re
text = 'abcdfghijk'
parser = re.search('a[b-f]*f', text)
print (parser)


print (parser.group())

#search RegEx
import re

text = "The ants go marching one by one"

strings = ['the', 'one']

for string in strings:
    match = re.search(string, text)
    if match:
        print('Found "{}" in "{}"'.format(string, text))
        text_pos = match.span()
        print(text[match.start():match.end()])
    else:
        print('Did not find "{}"'.format(string))

#compile RegEx
import re

text = "The ants go marching one by one"

strings = ['the', 'one']

for string in strings:
    regex = re.compile(string)
    match = re.search(regex, text)
    if match:
        print('Found "{}" in "{}"'.format(string, text))
        text_pos = match.span()
        print(text[match.start():match.end()])
    else:
        print('Did not find "{}"'.format(string))

#compilation tag
import re

def validate_input(input_email):

	
	re_compilation=re.compile(r"""
                           ^([a-z0-9_\.-]+)      #it will pick the first local part
                           @                     # will pick the @ sign
                            ([0-9a-z\.-]+)       # will pick the domain name
                           \.                    # will have single "."
                            ([a-z]{2,6})$        # it will pick the top level Domain (last part)    
                           """,
           re.VERBOSE)

	result=re_compilation.fullmatch(input_email)

	if result:
		print("{} is Valid.".format(input_email))
		
	else:
		print("{} is Invalid".format(input_email))


validate_input("name@gmail.com")
validate_input("educative@.com")

#multiple matches
import re
silly_string = "the cat in the hat"
pattern = "the"
print (re.findall(pattern, silly_string))

import re

silly_string = "the cat in the hat"
pattern = "the"

for match in re.finditer(pattern, silly_string):
    s = "Found '{group}' at {begin}:{end}".format(
        group=match.group(), begin=match.start(),
        end=match.end())
    print(s)
#r for backslash
testing_string = r'python "\"'
print(testing_string)

#Typing
def process_data(my_list: list, name: str) -> bool:
    return name in my_list

if __name__ == '__main__':
    my_list = ['Mike', 'Nick', 'Toby']
    print( process_data(my_list, 'Mike') )
    print( process_data(my_list, 'John') )

class Fruit:
    def __init__(self, name, color):
      self.name = name
      self.color = color


def salad(fruit_one: Fruit, fruit_two: Fruit) -> list:
    print(fruit_one.name)
    print(fruit_two.name)
    return [fruit_one, fruit_two]

if __name__ == '__main__':
    f = Fruit('orange', 'orange')
    f2 = Fruit('apple', 'red')
    salad(f, f2)

Animal = str #alias

def zoo(animal: Animal, number: int) -> None:
    print("The zoo has %s %s" % (number, animal))

if __name__ == '__main__':
    zoo('Zebras', 10)

#Type hints for overloading functions
from functools import singledispatch


@singledispatch
def add(a, b):
    raise NotImplementedError('Unsupported type')


@add.register(int)
def _(a: int, b: int) -> int:
    print("First argument is of type ", type(a))
    print(a + b)
    return a + b


@add.register(str)
def _(a: str, b: str) -> str:
    print("First argument is of type ", type(a))
    print(a + b)
    return a + b


@add.register(list)
def _(a: list, b: list) -> list:
    print("First argument is of type ", type(a))
    print(a + b)
    return a + b

if __name__ == '__main__':
   add(1, 2)
   add('Python', 'Programming')
   add([1, 2, 3], [5, 6, 7])

##built-ins
#any
print (any([0,0,0,1]))
#enumerate
my_string = 'abcdefg'
for pos, letter in enumerate(my_string):
    print (pos, letter)
#eval
var = 10
source = 'var * 2'
print (eval(source))
#filter
def less_than_ten(x):
    return x < 10

my_list = [1, 2, 3, 10, 11, 12]
for item in filter(less_than_ten, my_list):
    print(item)
#map
def doubler(x):
    return x * 2

my_list = [1, 2, 3, 4, 5]
for item in map(doubler, my_list):
    print(item)
#list comprehension
def doubler(x):
    return x * 2

my_list = [1, 2, 3, 4, 5]
print ([doubler(x) for x in my_list])

#zip to dict
keys = ['x', 'y', 'z']
values = [5, 6, 7]
my_dict = dict(zip(keys, values))
print (my_dict)

##Benchmark
#timeit
def my_function():
    try:
        1 / 0
    except ZeroDivisionError:
        pass

if __name__ == "__main__":
    import timeit
    setup = "from __main__ import my_function"
    print(timeit.timeit("my_function()", setup=setup))

#decorator timer
import random
import time

def timerfunc(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value
    return function_timer


@timerfunc
def long_runner():
    for x in range(5):
        sleep_time = random.choice(range(1,5))
        time.sleep(sleep_time)

if __name__ == '__main__':
    long_runner()

#timing using context manager
import random
import time

class MyTimer():

    def __init__(self):
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = 'The function took {time} seconds to complete'
        print(msg.format(time=runtime))


def long_runner():
    for x in range(5):
        sleep_time = random.choice(range(1,5))
        time.sleep(sleep_time)


if __name__ == '__main__':
    with MyTimer():
        long_runner()

##profilers
import cProfile
cProfile.run("[x for x in range(1500)]")

#hashing
import hashlib
md5 = hashlib.md5()

md5.update(b'Python rocks!')
print (md5.digest())

import hashlib
sha = hashlib.sha1(b'Hello Python').hexdigest()
print (sha)

#pass hashing
from Crypto.Protocol.KDF import scrypt

password = b'password@educative'
salt=b'educative_salt'
key = scrypt(password, salt, 16, N=2**14, r=8, p=1)
print(key)

import bcrypt
 
password = b'password@educative'
 
salt_generation = bcrypt.gensalt(10)
print("Randomly generated salt after 10 rounds: ")
print(salt_generation)

password_hash = bcrypt.hashpw(password, salt_generation)
print("Password hashed after random generation of salt:")
print(password_hash)

password_match = bcrypt.checkpw(password, password_hash)
print("if the password matches with already hashed password:")
print(password_match)

##DB
import sqlalchemy as sal
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://educative:secret@localhost:5432/test")
conn = engine.connect()
cursor = conn_sql.cursor()
print(engine)
# execute a query
cursor.execute("SELECT * FROM table_name;")
row = cursor.fetchone()
print(row)
# close your cursor and connection
cursor.close()
conn.close()

##super
class MyParentClass:
    def __init__(self, course, institute):
        self.course_name = course
        self.institute_name = institute

    def printname(self):
         print(self.course_name, self.institute_name)

class SubClass(MyParentClass):
    def __init__(self, course, institute):
        super().__init__(course, institute)

subclass = SubClass("Python201", "Educative")
subclass.printname()

##validate using descriptor
from weakref import WeakKeyDictionary

class Drinker:
    def __init__(self):
        self.req_age = 21
        self.age = WeakKeyDictionary()

    def __get__(self, instance_obj, objtype):
        return self.age.get(instance_obj, self.req_age)

    def __set__(self, instance, new_age):
        if new_age < 21:
            msg = '{name} is too young to legally imbibe'
            raise Exception(msg.format(name=instance.name))
        self.age[instance] = new_age
        print('{name} can legally drink in the USA'.format(
            name=instance.name))

    def __delete__(self, instance):
        del self.age[instance]


class Person:
    drinker_age = Drinker()

    def __init__(self, name, age):
        self.name = name
        self.drinker_age = age


p = Person('Miguel', 30)
p = Person('Niki', 13)

##scope
#nonlocal
def counter():
    num = 0
    def incrementer():
        num += 1 # referenced before it is assigned, must use nonlocal num first
        return num
    return incrementer

##Testing
#doctest
def add(a, b):
    """
    Return the addition of the arguments: a + b

    add(1, 2)
    #3
    add(-1, 10)
    #9
    add('a', 'b')
    #'ab'
    add(1, '2')
    #Traceback (most recent call last):
    #  File "test.py", line 17, in <module>
    #    add(1, '2')
    #  File "test.py", line 14, in add
    #    return a + b
    #TypeError: unsupported operand type(s) for +: 'int' and 'str'
    """
    return a + b

print(add(1, 2))
print(add(-1, 10))
print(add('a', 'b'))
print(add(1, '2'))

if __name__ == '__main__':
    import doctest
    doctest.testmod()

#using flag
"""
print(list(range(100))) # doctest: +ELLIPSIS
#[0, 1, ..., 98, 99]

class Dog: pass
Dog() #doctest: +ELLIPSIS
#<__main__.Dog object at 0x...>
"""

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
#unittest
import os
# import simple_db
import sqlite3
import unittest

class TestMusicDatabase(unittest.TestCase):
    """
    Test the music database
    """

    def setUp(self):
        """
        Setup a temporary database
        """
        conn = sqlite3.connect("mydatabase.db")
        cursor = conn.cursor()

        # create a table
        cursor.execute("""CREATE TABLE albums
                          (title text, artist text, release_date text,
                           publisher text, media_type text)
                       """)
        # insert some data
        cursor.execute("INSERT INTO albums VALUES "
                       "('Glow', 'Andy Hunter', '7/24/2012',"
                       "'Xplore Records', 'MP3')")

        # save data to database
        conn.commit()

        # insert multiple records using the more secure "?" method
        albums = [('Exodus', 'Andy Hunter', '7/9/2002',
                   'Sparrow Records', 'CD'),
                  ('Until We Have Faces', 'Red', '2/1/2011',
                   'Essential Records', 'CD'),
                  ('The End is Where We Begin', 'Thousand Foot Krutch',
                   '4/17/2012', 'TFKmusic', 'CD'),
                  ('The Good Life', 'Trip Lee', '4/10/2012',
                   'Reach Records', 'CD')]
        cursor.executemany("INSERT INTO albums VALUES (?,?,?,?,?)",
                           albums)
        conn.commit()

    def tearDown(self):
        """
        Delete the database
        """
        os.remove("mydatabase.db")

    def test_updating_artist(self):
        """
        Tests that we can successfully update an artist's name
        """
        simple_db.update_artist('Red', 'Redder')
        actual = simple_db.select_all_albums('Redder')
        expected = [('Until We Have Faces', 'Redder',
                    '2/1/2011', 'Essential Records', 'CD')]
        self.assertListEqual(expected, actual)

    def test_artist_does_not_exist(self):
        """
        Test that an artist does not exist
        """
        result = simple_db.select_all_albums('Redder')
        self.assertFalse(result)

    @unittest.skip('Skip this test')
    def test_add_strings(self):
        """
        Test the addition of two strings returns the two string as one
        concatenated string
        """
        result = mymath.add('abc', 'def')
        self.assertEqual(result, 'abcdef')

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_adding_on_windows(self):
        result = mymath.add(1, 2)
        self.assertEqual(result, 3)

#suite
import unittest

from test_mymath import TestAdd


def my_suite():
    suite = unittest.TestSuite()
    result = unittest.TestResult()
    suite.addTest(unittest.makeSuite(TestAdd))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))

my_suite()
#unittest doctest
import doctest
import my_docs
import unittest

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(my_docs))
    return tests

#mock
from unittest.mock import Mock
class TestClass():
    pass

cls = TestClass()
cls.method = Mock(return_value='mocking is fun')
cls.method(1, 2, 3)
#'mocking is fun'

cls.method.assert_called_once_with(1, 2, 3)
cls.method(1, 2, 3)
#'mocking is fun'

cls.method.assert_called_once_with(1, 2, 3)
#Traceback (most recent call last):
# File "/usercode/__ed_file.py", line 14, in <module>
# cls.method.assert_called_once_with(1, 2, 3)
# File "/usr/lib/python3.5/unittest/mock.py", line 804, in assert_called_once_with
# raise AssertionError(msg)
# AssertionError: Expected 'mock' to be called once. Called 2 times.

cls.other_method = Mock(return_value='Something else')
cls.other_method.assert_not_called()

#side effect
from unittest.mock import Mock


def my_side_effect():
    print('Updating database!')

def main():
    mock = Mock(side_effect=my_side_effect)
    mock()

if __name__ == '__main__':
    main()

#autospeccing
from unittest.mock import create_autospec
def add(a, b):
    return a + b

mocked_func = create_autospec(add, return_value=10)
print (mocked_func(1, 2))
#10

mocked_func(1, 2, 3)
#Traceback (most recent call last):
# File "/usercode/__ed_file.py", line 9, in <module>
# mocked_func(1, 2, 3)
# File "<string>", line 2, in add
# File "/usr/lib/python3.5/unittest/mock.py", line 183, in checksig
# sig.bind(*args, **kwargs)
# File "/usr/lib/python3.5/inspect.py", line 2918, in bind
# return args[0]._bind(args[1:], kwargs)
# File "/usr/lib/python3.5/inspect.py", line 2839, in _bind
# raise TypeError('too many positional arguments') from None
#TypeError: too many positional arguments

#patch
import urllib.request

def read_webpage(url):
    response = urllib.request.urlopen(url)
    return response.read()

# import webreader

from unittest.mock import patch


@patch('urllib.request.urlopen')
def dummy_reader(mock_obj):
    result = read_webpage('https://www.google.com/') #webreader.read_webpage('https://www.google.com/')
    mock_obj.assert_called_with('https://www.google.com/')
    print(result)

if __name__ == '__main__':
    dummy_reader()

##asyncio
import asyncio

async def my_coro():
    await func()

#coroutine
import aiohttp
import asyncio
import async_timeout
import os

import time

async def download_coroutine(session, url):
    with async_timeout.timeout(1000):
        async with session.get(url) as response:
            filename = os.path.basename(url)
            with open(filename, 'wb') as f_handle:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f_handle.write(chunk)
                await response.release()
                return response.status == 200

async def main(loop):
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]

    async with aiohttp.ClientSession(loop=loop) as session:
        for url in urls:
            print(await download_coroutine(session, url))
            


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))

#scheduling calls
import asyncio
import functools


def event_handler(loop, stop=False):
    print('Event handler called')
    if stop:
        print('stopping the loop')
        loop.stop()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.call_soon(functools.partial(event_handler, loop))
        print('starting event loop')
        loop.call_soon(functools.partial(event_handler, loop, stop=True))

        loop.run_forever()
    finally:
        print('closing event loop')
        loop.close() 

loop.call_later(1, event_handler, loop)
current_time = loop.time()
loop.call_at(current_time + 300, event_handler, loop)

##threading
import logging
import threading

def get_logger():
    logger = logging.getLogger("threading_example")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("threading.log")
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def doubler(number, logger):
    """
    A function that can be used by a thread
    """
    logger.debug('doubler function executing')
    result = number * 2
    logger.debug('doubler function ended with: {}'.format(
        result))


if __name__ == '__main__':
    logger = get_logger()
    thread_names = ['Mike', 'George', 'Wanda', 'Dingbat', 'Nina']
    for i in range(5):
        my_thread = threading.Thread(
            target=doubler, name=thread_names[i], args=(i,logger))
        my_thread.start()
        print(thread_names[i])

##lock
import threading

total = 0
lock = threading.Lock()

def update_total(amount):
    """
    Updates the total by the given amount
    """
    global total
    lock.acquire()
    try:
        total += amount
    finally:
        lock.release()
    print (total)

if __name__ == '__main__':
    for i in range(10):
        my_thread = threading.Thread(
            target=update_total, args=(5,))
        my_thread.start()

import threading

total = 0
lock = threading.RLock()

def do_something():

    with lock:
        print('Lock acquired in the do_something function')
    print('Lock released in the do_something function')

    return "Done doing something"

def do_something_else():
    with lock:
        print('Lock acquired in the do_something_else function')
    print('Lock released in the do_something_else function')

    return "Finished something else"


def main():
    with lock:
        result_one = do_something()
        result_two = do_something_else()

    print (result_one)
    print (result_two)

if __name__ == '__main__':
    main()

#semaphores
import threading

semaphore = threading.Semaphore(2)


def func():
    semaphore.acquire()
    print(semaphore)
#Acquiring lock
    print("%s Lock acquired." % (threading.current_thread().name)) 
#Thread access decremented
    print("Available threads access: ", semaphore._value) 
    semaphore.release()
#Releasing lock
    print("%s Lock released." % (threading.current_thread().name)) 
#Thread access incremented
    print("Available threads access: ", semaphore._value) 


thread1 = threading.Thread(target=func)
thread2 = threading.Thread(target=func)
thread3 = threading.Thread(target=func)

thread1.start()
thread2.start()
thread3.start()
print("Main thread exited.", threading.main_thread())

#barrier
import time
from threading import Barrier, Thread

barrier = Barrier(2)

def wait_at_barrier (name, time_to_sleep):
    for i in range(10):
        print (name, "executing.")
        time.sleep(time_to_sleep)
        print (name, "waiting at barrier.")
        barrier.wait()
    print (name, "is finished.")

thread1 = Thread(target=wait_at_barrier, args=["thread1", 3])
thread2 = Thread(target=wait_at_barrier, args=["thread2", 10])

thread1.start()
thread2.start()
time.sleep(11)
print("Aborting barrier")
barrier.abort()

#queue
import threading

from queue import Queue


def creator(data, q):
    """
    Creates data to be consumed and waits for the consumer
    to finish processing
    """
    print('Creating data and putting it on the queue')
    for item in data:
        evt = threading.Event()
        q.put((item, evt))

        print('Waiting for data to be doubled')
        evt.wait()


def my_consumer(q):
    """
    Consumes some data and works on it

    In this case, all it does is double the input
    """
    while True:
        data, evt = q.get()
        print('data found to be processed: {}'.format(data))
        processed = data * 2
        print(processed)
        evt.set()
        q.task_done()


if __name__ == '__main__':
    q = Queue()
    data = [5, 10, 13, -1]
    thread_one = threading.Thread(target=creator, args=(data, q))
    thread_two = threading.Thread(target=my_consumer, args=(q,))
    thread_one.start()
    thread_two.start()

    q.join()

#multiprocessing
import os

from multiprocessing import Process, current_process


def doubler(number):
    """
    A doubling function that can be used by a process
    """
    result = number * 2
    proc_name = current_process().name
    print('{0} doubled to {1} by: {2}'.format(
        number, result, proc_name))


if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25]
    procs = []
    proc = Process(target=doubler, args=(5,))

    for index, number in enumerate(numbers):
        proc = Process(target=doubler, args=(number,))
        procs.append(proc)
        proc.start()

    proc = Process(target=doubler, name='Test', args=(2,))
    proc.start()
    procs.append(proc)

    for proc in procs:
        proc.join()

#lock
from multiprocessing import Process, Lock


def printer(item, lock):
    """
    Prints out the item that was passed in
    """
    lock.acquire()
    try:
        print(item)
    finally:
        lock.release()

if __name__ == '__main__':
    lock = Lock()
    items = ['tango', 'foxtrot', 10]
    for item in items:
        p = Process(target=printer, args=(item, lock))
        p.start()

#logging
import logging
import multiprocessing

from multiprocessing import Process, Lock


def printer(item, lock):
    """
    Prints out the item that was passed in
    """
    lock.acquire()
    try:
        print(item)
    finally:
        lock.release()

if __name__ == '__main__':
    lock = Lock()
    items = ['tango', 'foxtrot', 10]
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    for item in items:
        p = Process(target=printer, args=(item, lock))
        p.start()

#pool
from multiprocessing import Pool


def doubler(number):
    return number * 2

if __name__ == '__main__':
    numbers = [5, 10, 20]
    pool = Pool(processes=3)
    print(pool.map(doubler, numbers))

from multiprocessing import Pool


def doubler(number):
    return number * 2

if __name__ == '__main__':
    pool = Pool(processes=3)
    result = pool.apply_async(doubler, (25,))
    print(result.get(timeout=1))

#communicating
from multiprocessing import Process, Queue


sentinel = -1

def creator(data, q):
    """
    Creates data to be consumed and waits for the consumer
    to finish processing
    """
    print('Creating data and putting it on the queue')
    for item in data:

        q.put(item)


def my_consumer(q):
    """
    Consumes some data and works on it

    In this case, all it does is double the input
    """
    while True:
        data = q.get()
        print('data found to be processed: {}'.format(data))
        processed = data * 2
        print(processed)

        if data is sentinel:
            break


if __name__ == '__main__':
    q = Queue()
    data = [5, 10, 13, -1]
    process_one = Process(target=creator, args=(data, q))
    process_two = Process(target=my_consumer, args=(q,))
    process_one.start()
    process_two.start()

    q.close()
    q.join_thread()

    process_one.join()
    process_two.join()

##futures
import os
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


def downloader(url):
    """
    Downloads the specified URL and saves it to disk
    """
    req = urllib.request.urlopen(url)
    filename = os.path.basename(url)
    ext = os.path.splitext(url)[1]
    if not ext:
        raise RuntimeError('URL does not contain an extension')

    with open(filename, 'wb') as file_handle:
        while True:
            chunk = req.read(1024)
            if not chunk:
                break
            file_handle.write(chunk)
    msg = 'Finished downloading {filename}'.format(filename=filename)
    return msg


def main(urls):
    """
    Create a thread pool and download specified urls
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        return executor.map(downloader, urls, timeout=60)

if __name__ == '__main__':
    urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
            "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]
    results = main(urls)
    for result in results:
        print(result)
