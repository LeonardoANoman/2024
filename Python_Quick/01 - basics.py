# This is a comment line

"""
This are
multiple
comments
"""

print("Output a message\n")

# Types and Variables

string = "string"
integer = 3
float_number = 2.4
boolean = True 
none_type = None


# Operations

addition = 1 + 1.1
subtraction = 2 - 1
multiplication = 2 * 4
division = 10 / 2.5
modulus = 5 % 2 
exponentiation = 2 ** 3
floor_division = 10//3

add_and_assign = 0
add_and_assign += 5

# Comparision

zero = 0

zero == 0
zero != 1
zero >= -1
zero > -2
zero <= 1
zero < 1

zero and 1 == 0
zero or 1 == 1
not(zero or 1) == False

# Identity

zero is 0
zero is not 1

# Control structures

if zero > -1:
    print("Zero is greater than -1\n")
elif zero < -1:
    print("Zero is less than -1\n")
else:
    print("Zero is equal to -1\n")
    
print("While\n") 

while zero < 5:
    print(zero)
    zero += 1

print("")
print("for\n") 

for i in range(5):
    print(i)
   
print("") 
print("for start, stop\n") 

for i in range(-3,4):
    print(i)
    
print("")
print("for start, stop, sep\n") 

for i in range(0,10,2):
    print(i)
print("")


# Data Structures

random_list = ["This is a list", 3, False]
print("")

for i in random_list:
    print(i)
print("")

random_tuple = ("This is a tuple", "It's immutable")
print(random_tuple)
print("")

random_set = set([1,2,2,3,3,3])
print(random_set)
print("")

random_dict = {'key':'value'}
print(random_dict)
print("")

# Common Data Structure Operations

random_list.append("On the last index")
print(random_list[-1])
print("")

print(random_tuple[0])
print("")

print(random_set.pop())
print("")

print(random_dict.get('key'))
print("")


# Functions

def first_function(x):
    print(x)
    
first_function("Calling my function")
print("")

# List Comprehension

squares = [x ** 2 for x in range(5)]
print(squares)
print("")

# Generators

def square_numbers(n):
    for i in range(n):
        yield i**3

cubes = square_numbers(5)
for cube in cubes:
    print(cube)
print("")

# Decorators

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
print("")


# Object Oriented Programming

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        print(f"{self.name} is barking!")

my_dog = Dog("Buddy", 3)
print(my_dog.name)  
print(my_dog.age)   
my_dog.bark()      
print("")

# Inheritance

class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  
print(cat.speak())  
print("")

# Polymorphism

class Bird:
    def fly(self):
        print("Bird is flying")

class Airplane:
    def fly(self):
        print("Airplane is flying")

def let_it_fly(flying_object):
    flying_object.fly()

bird = Bird()
airplane = Airplane()

let_it_fly(bird)      
let_it_fly(airplane)   
print("")

# Encapsulation

class Person:
    def __init__(self, name, age):
        self.name = name
        self.__age = age 

    def get_age(self):
        return self.__age

    def set_age(self, age):
        if age > 0:
            self.__age = age
        else:
            print("Age must be positive")

person = Person("Alice", 30)
print(person.get_age())  
person.set_age(35)
print(person.get_age())  
person.set_age(-5)      
print("") 

# Exceptions

try:
    result = 10/0
    print(10/0)
except ZeroDivisionError as e:
    print(f"Error: {e}")
else:
    print(f"Result: {result}")
finally:
    print("Execution complete")
print("")

# Define a custom exception class
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Custom Exception
def check_positive_number(number):
    if number < 0:
        raise CustomError("Negative numbers are not allowed")
    else:
        print(f"{number} is a positive number")

try:
    check_positive_number(-5)
except CustomError as e:
    print(f"CustomError: {e}")
else:
    print("No error occurred")
finally:
    print("Execution complete")
print("")

# Writing, Appending and Reading a File

with open('example.txt', 'w') as file:
    file.write("Hello, world!\n")
    file.write("This is a second line.\n")
    
with open('example.txt', 'a') as file:
    file.write("This line is appended.\n")

with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
print("")

