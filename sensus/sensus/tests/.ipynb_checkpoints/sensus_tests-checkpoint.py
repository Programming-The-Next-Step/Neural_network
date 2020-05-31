from sensus import utilities
from unittest import TestCase
from urllib.request import urlretrieve
import numpy as np
import random
import os
import gzip 

# Load the data for all test functions 

x_train = utilities.load_mnist_images('train-images-idx3-ubyte.gz')
y_train = utilities.load_mnist_labels('train-labels-idx1-ubyte.gz')
x_test = utilities.load_mnist_images('t10k-images-idx3-ubyte.gz')
y_test = utilities.load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
stored = utilities.store_data(x_train, y_train, x_test, y_test)
training_dat = stored[0]
test_dat = stored[1]
formatted = utilities.format_data(training_dat, test_dat)
    
training_data = list(formatted[0])
test_data = list(formatted[1])

# define the network
net = utilities.Network(sizes = [784, 30, 10])

def test_feedforward(): 
    s = net.feedforward(a = training_data[0][0])
    assert isinstance(s, np.ndarray) and len(s) == net.sizes[2] # the output is a Numpy array and it's length equals the size of the output layer
    
def test_backprop():
    s = net.backprop(x = training_data[0][0], y = training_data[0][1])
    assert isinstance(s, tuple) and len(s) == 2 # the output is a tuple containing two arrays; a) the gradients of the weights and b) the gradients of the biases
    
def test_SGD():
    s = net.SGD(training_data = training_data, epochs = 30, mini_batch_size = 10, eta = 3.0, test_data = test_data)
    assert isinstance(s, tuple) and len(s[0]) == s[1] # the output is a tuple. The first element is the vector of correct classifications. The second element is the number of epochs. The two must be equal. 