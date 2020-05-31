from urllib.request import urlretrieve
import numpy as np
import random
import os
import gzip

# A function to load the MNIST data 

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)
    
# We then define functions for loading MNIST images and labels.
# For convenience, they also download the requested files if needed.

def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
            
    # Read the inputs in Yann LeCun's binary format.
    
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
            
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    
    data = data.reshape(-1, 1, 28, 28)
    
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    
    return data / np.float32(256)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
            
    # Read the labels in Yann LeCun's binary format.
    
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
            
    # The labels are vectors of integers now, that's exactly what we want.
    
    return data
    
# The following functions prepare the data to be used in the code. 

def store_data(x_train, y_train, x_test, y_test):
    training_dat = (x_train, y_train) # x_train has 60,000 matrices (28x28), y_has 60,000 digits
    test_dat = (x_test, y_test)
    stored = (training_dat, test_dat)
    return stored

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def format_data(training_dat, test_dat):
    training_inputs = [np.reshape(x, (784, 1)) for x in training_dat[0]]
    training_results = [vectorized_result(y) for y in training_dat[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in test_dat[0]]
    test_data = zip(test_inputs, test_dat[1])
    return (training_data, test_data)

# Load the data 

x_train = load_mnist_images('train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

# Prepare the data to be used in the programme

stored = store_data(x_train, y_train, x_test, y_test)
training_dat = stored[0]
test_dat = stored[1]
formatted = format_data(training_dat, test_dat)
training_data = list(formatted[0])
test_data = list(formatted[1])

# %load sensus.py

"""
sensus.py
~~~~~~~~~~
A simple neural network that recognizes hand written digits. 
It is based on Michael Nielsen's code but I did not divide the 
data into validation and test data sets. Also, the backpropagation 
algorithm uses the matrix approach by Hind Sellouk, rather than 
looping over the training examples in a minibatch, which is what 
Nielsen's code does. I have adapted the code to Python 3.
"""

class Network:

    # initializaing the weights and the biases with random numbers 
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        """
        Stochastic gradient descent with backpropagation. After training the 
        network with stochastic gradient descent (SGD) using backpropagation, 
        returns a vector of the total number of correct classifications per epoch. 
        Also prints the proportion of the number of correct classifications for each epoch.
        """
        epochs == epochs # necessary for the tests
        
        #training_data = list(training_data) # necessary because the data sets are in zip format
        n = len(training_data)

        #test_data = list(test_data)
        n_test = len(test_data)
        
        corrects = [] # a vector to store the correct classifactions per epoch (used in testing)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            corrects.append(self.evaluate(test_data)) # checking against the test data
                                                      # returning a vector with correct classifications for each epoch
            
            print("Epoch {0}: {1} / {2}".format(
                j, self.evaluate(test_data), n_test))
                
        return (corrects, epochs)

    def update_mini_batch(self, mini_batch, eta):
        """ 
        Updates the weights and the biases for minibatches, 
        using the gradients computed with backpropagation. 
        """
        matrix_X = mini_batch[0][0]
        matrix_Y = mini_batch[0][1]
        
        # create matrix_X of examples and a matrix_Y of labels
        
        for x,y in mini_batch[1:]:
            matrix_X = np.concatenate((matrix_X, x), axis = 1)
            matrix_Y = np.concatenate((matrix_Y, y), axis = 1)

        nabla_b, nabla_w = self.backprop(matrix_X, matrix_Y)

        self.weights = [w - (eta / len(mini_batch)) * nw
                    for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                   for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Returns the gradients of the weights and the biases"""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        
        activation = x
        activations = [x] # list to store all the activation matrices, layer by layer
        zs = [] # list to store all the "sum of weighted inputs z" matrices, layer by layer
        i = 0
        
        for b, w in zip(self.biases, self.weights):
            w = np.insert(w, 0, b.transpose(), axis = 1) #insert the vector of biases on the first column of the weight matrix
            i += 1
            activation = np.insert(activation, 0, np.ones(activation[0].shape), 0)#insert ones on the first line of the matrix of activations
            z = np.dot(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = np.expand_dims(np.sum(delta,axis = 1), axis = 1)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.expand_dims(np.sum(delta, axis = 1), axis = 1)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def feedforward(self, a):
        """ 
        Returns a vector of outputs, each value being the 
        activation of an output neuron.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def evaluate(self, test_data):
        """
        Returns the total number of images that were classified
        correctly in each epoch.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Returns the partial derivative of the cost function
        with respect to the output activations.
        """
        return (output_activations - y)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))