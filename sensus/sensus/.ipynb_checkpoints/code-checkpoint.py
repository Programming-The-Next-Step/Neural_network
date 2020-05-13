# We need three layers. Input neurons are the intensities of the pixels: i x j pixels means i x j input neurons.
# 1) 28 x 28 = 784 dimensional input vector, which is the image of a single digit. Each value in this vector 
# indicates what shade of grey a pixel is between white and black. y is a 10 dimensional output vector, one value 
# for each digit between 0-9 if the network decides on 5 for a given training image, the output vector will be 
# y(x) = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0) transposed. 

import numpy as np

class Network:
    
    def __init__(self, sizes): # 'sizes' is a list, with each element being the number of neurons in a layer
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]] # a vector of initial biases, excluding the input layer
        self.weights = [np.random.randn(j, k) for k, j in zip(sizes[:-1], sizes[1:])] # j and k were swapped deliberately

class Neuron: 
    "A simple neuron" 
    
    def z(self, w, a, b): # w is a matrix of weights between two layers, 
                          # a is vector with the activations of the preceding layer, 
                          # b is a vector with the biases of the computed layer 
        """ The input of the neuron """
        z = w * a + b # z is the resulting vector
        return z 
    
    def sigmoid(self, z):
        """ The output of the neuron """
        output = 1 / (1 + np.exp(-z)) # Numpy applies the sigmoid function elementwise
        return output # this is the output vector, which will be the activation vector for the next layer
    

