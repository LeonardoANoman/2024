# MySQL

import mysql.connector
from mysql.connector import Error

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

host = "your_host"
user = "your_username"
password = "your_password"
database = "your_database"

connection = create_connection(host, user, password, database)

create_table_query = """
CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT, 
  name TEXT NOT NULL, 
  age INT, 
  gender TEXT, 
  nationality TEXT, 
  PRIMARY KEY (id)
) ENGINE = InnoDB
"""
execute_query(connection, create_table_query)

insert_user_query = """
INSERT INTO users (name, age, gender, nationality) VALUES
('James', 25, 'male', 'USA'),
('Leila', 32, 'female', 'France'),
('Brigitte', 35, 'female', 'England')
"""
execute_query(connection, insert_user_query)

select_users_query = "SELECT * FROM users"
users = read_query(connection, select_users_query)

for user in users:
    print(user)
    

# PostgreSQL

import psycopg2
from psycopg2 import OperationalError

def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_query(connection, query):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Query executed successfully")
    except OperationalError as e:
        print(f"The error '{e}' occurred")

def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except OperationalError as e:
        print(f"The error '{e}' occurred")

database = "your_database"
user = "your_username"
password = "your_password"
host = "your_host"
port = "your_port"

connection = create_connection(database, user, password, host, port)

create_table_query = """
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY, 
  name TEXT NOT NULL, 
  age INTEGER, 
  gender TEXT, 
  nationality TEXT
)
"""
execute_query(connection, create_table_query)

insert_user_query = """
INSERT INTO users (name, age, gender, nationality) VALUES
('James', 25, 'male', 'USA'),
('Leila', 32, 'female', 'France'),
('Brigitte', 35, 'female', 'England')
"""
execute_query(connection, insert_user_query)

select_users_query = "SELECT * FROM users"
users = read_query(connection, select_users_query)

for user in users:
    print(user)
