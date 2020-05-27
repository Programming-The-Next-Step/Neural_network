### This week was about learning the backpropagation algorithm and implementing the equations in code. I have two short functions left. I will finish those tomorrow and test the network. On Friday I will change the code. Not the content, just the way it is coded. That's because most of it is from a book. I already changed the parts that were meant for Python 2.x. I also changed variable names as some of them did not make sense to me, and almost all the docstring.  

import random 
import numpy as np

class Network:
    
    # Initializing the weights and the biases
    
    def __init__(self, sizes): # 'sizes' is a list, with each element being the number of neurons in a layer
        self.n_layers = len(sizes)
        self.sizes = sizes 
        self.weights = [np.random.randn(j, k) for k, j in zip(sizes[:-1], sizes[1:])] # j and k were swapped deliberately
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]] # a vector of initial biases, excluding the input layer
    
    # The activation function that computes the input of a neuron as the weighted sum of the previous neurons' outputs, 
    # the weights and the biases
    
    def activation(self, a): # 'a' is a vector of outputs from the previous layer (i.e., input of the current layer)
        """ The activation of a neuron """
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b 
        return z
    
    # The 'activation function' sigmoid, which compresses the activation value to a value between 0 and 1.
    
    def sigmoid(self, z): 
        """ The output of a neuron """
        return 1 / (1 + np.exp(-z))
   
    # We enter the activation value of the neurons into the sigmoid function to get the outputs. 
    # This specific function will compute the output of the neural network. 
    
    def output(self, a):
        return sigmoid(activation(a))
    
    ## The following functions will enable the output function to give a correct value. Our goal is to find the weights
    ## and the biases that will lead to the correct output. To find this, we need to find how much the cost function 
    ## (i.e., the difference between the network's output and the correct output, as provided by the training data) 
    ## would change based on changes in the weights and the biases. 
    
    # We start by finding the change in the cost function based on the changes in the output activations. 
    # How less wrong would we be if the final output activations were different? This information is the first 
    # step towards finding the change in the cost function for weights and biases, because it will be communicated
    # back to the previous layers by 'backpropagation'. 
    
    def cost_output_derivative(self, output_activations, y): # output_activations are created in backpropagation
        """ Returns the partial derivates of the cost function with respect to all the output activations. """
        return output_activations - y
    
    def sigmoid_prime(z): # multiplying this with the derivative of the cost function we get an error term, which 
                          # tells us how much the output would change depending on the output activations. 
        """ Returns the derivative of the cost function. """
        return sigmoid(z) * (1-sigmoid(z))
    
    def backpropagation(self, x, y):
        """ Returns the partial derivative of the cost function with respect to the weights and the biases. 
        The output is a tuple (grad_w, grad_b). The first element is a list of vectors, each vector containing 
        the partial derivatives of the cost function with respect to the weights in a single layer. The second 
        is the same with respect to biases. """
    
    grad_w = [np.zeros(w.shape) for w in self.weights]
    grad_b = [np.zeros(b.shape) for b in self.biases]
    
    # first the information (the activations / outputs) is propagated forward via the weights and the biases 
    activation = x
    activations = [x] # a list of vectors that will store all the activations in all the layers 
    zs = [] # the same for the weighted sums (the activation of an input neuron which has not been into the sigmoid yet)
    for w, b in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    
    # propagating backwards: computing the error term
    # calculated from the last layer, so index = -1
    error = self.cost_output_derivative(activations[-1], y) * sigmoid_primte(zs[-1])
    grad_w[-1] = np.dot(error, activations[-2].T)
    grad_b[-1] = error 
    
    for l in range(2, self.n_layers):
        z = zs[-l]
        sig_prime = sigmoid_prime(z)
        error = np.dot(self.weights[-l + 1].T, error) * sig_prime
        grad_w[-l] = np.dot(delta, activations[-l - 1].T))
        grad_b[-l] = delta
    return(grad_w, grad_b)
    
    def update_mini_batch(self, mini_batch, eta): # eta is just a constant that determines rate of learning. Do not concern
                                                  # yourself with it. The 'mini_batch' needs some explanation. The data 
                                                  # consist of a list of tuples. The tuples (x, y) contain the inputs and the 
                                                  # desired outputs. A mini batch is a randomly selected sub sample of the data. 
                                                  # The gradient descent is calculated over these samples and then averaged. 
                                                  # It speeds up the process while giving accurate estimations of the gradient. 
                                                  # This function updates the weights and the biases for a single batch. It may 
                                                  # seem a bit mysterious at first because there is no code that actually 
                                                  # calculates the gradient. I will add a function for that in the following days. 
                                                  # It will be contained in the backprop() function, which is in this chunk.
                                                  # I have not finished learning the back propagation algorithm yet but its 
                                                  # results should be used as specified here.
                                                    
        grad_b = [np.zeros(b.shape) for b in self.biases] # vectors that will store the gradient for weights and biases
        grad_w = [np.zeros(w.shape) for w in self.weights] # initially both are full of zeros. We will update them below.
                                                           
        for x, y in mini_batch: 
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [nb+dnb for nb, dnb in zip(grad_b, delta_grad_b)] # accumulating change
            grad_w = [nw+dnw for nw, dnw in zip(grad_w, delta_grad_w)]

        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, grad_b)] # subtract the average change in the 
                                                                                         # biases from the biases to update the biases                                                                      
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, grad_w)] # same for the weights 




