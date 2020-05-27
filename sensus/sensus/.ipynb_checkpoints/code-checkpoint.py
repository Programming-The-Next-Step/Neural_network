import numpy as np

class Network:
    
    def __init__(self, sizes): # 'sizes' is a list, with each element being the number of neurons in a layer
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]] # a vector of initial biases, excluding the input layer
        self.weights = [np.random.randn(j, k) for k, j in zip(sizes[:-1], sizes[1:])] # j and k were swapped deliberately
        
    def sigmoid(self, z): 
        """ The output of a neuron """
        return 1 / (1 + np.exp(-z)) # the sigmoid function transforms the output into a number between 0 and 1, which becomes
                                 # input to the next layer
    
    def activation(self, a): # a is the vector of outputs from the previous layer (i.e., input of the current layer)
        """ The activation of a neuron """
        for w, b in zip(self.weights, self.biases):
            z = w * a + b # the activations of the neurons in the previous layer (i.e., inputs) are multiplied by the 
                          # weights and added to the biases of the neurons in the current layer
        return z
    
    def output(self, a):
        return sigmoid(activation(a))
    
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
                                                                                         # biases from the biases to update biases                                                                      
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, grad_w)] # same for the weights 

