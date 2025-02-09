import math

def multiply(x,y): 
    return x * y

def divide(x,y):
    if y == 0:
        raise ValueError('Can not divide by zero!')
    return x / y

def sqrt(x):
    if x < 0:
        raise ValueError('Can not take square root of negative number!')
    return math.sqrt(x)