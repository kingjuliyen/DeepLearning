
import numpy as np

def sigmoid(v):
    return (1 / (1 + np.exp(-v)))

def log(v):
    return np.log(v)

def sum(v):
    return np.sum(v)
