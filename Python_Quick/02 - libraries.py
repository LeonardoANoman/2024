import os

cwd = os.getcwd()
print(f"Current working directory: {cwd}")
files = os.listdir(cwd)
print("Files and directories in current directory:", files)
os.mkdir('new_directory')
os.rmdir('new_directory')
print("")

import sys

print("Command line arguments:", sys.argv)

import math

sqrt_value = math.sqrt(16)
print(f"Square root of 16: {sqrt_value}")

cos_value = math.cos(math.pi / 4)
print(f"Cosine of pi/4: {cos_value}")

factorial_value = math.factorial(5)
print(f"Factorial of 5: {factorial_value}")
print("")

from datetime import datetime, timedelta

now = datetime.now()
print(f"Current date and time: {now}")

formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"Formatted date and time: {formatted_date}")

future_date = now + timedelta(days=10)
print(f"Date 10 days from now: {future_date}")
print("")

import random

rand_num = random.randint(1, 100)
print(f"Random number between 1 and 100: {rand_num}")

rand_float = random.random()
print(f"Random float between 0 and 1: {rand_float}")

choices = ['apple', 'banana', 'cherry']
rand_choice = random.choice(choices)
print(f"Random choice from list: {rand_choice}")
print("")

import json

data = {"name": "John", "age": 30, "city": "New York"}
json_data = json.dumps(data)
print(f"JSON string: {json_data}")

python_data = json.loads(json_data)
print(f"Python dictionary: {python_data}")
print("")

import re

pattern = r'\bword\b'
text = "This is a word in a sentence."
match = re.search(pattern, text)
if match:
    print("Pattern found:", match.group())

matches = re.findall(r'\d+', 'There are 123 apples and 456 oranges')
print("All matches:", matches)

replaced_text = re.sub(r'\d+', 'number', 'There are 123 apples and 456 oranges')
print("Replaced text:", replaced_text)
print("")