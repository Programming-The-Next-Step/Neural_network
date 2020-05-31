# Package name: Sensus
## Author: Eren Asena 

***Sensus is a basic artificial neural network (ANN) that can recognize hand written digits.***

## An explanation of neural networks
A neural network is a computing system that can learn from data. First, training data are provided to the neural network. 
Next, these are converted into an output (e.g., producing the output 7 in response to a handwritten 4). Then, the neural 
network receives feedback about its performance. This feedback is given by communicating the discrepancy between its decision 
and the correct answer back to the network. The network minimizes this discrepancy by adjusting its outputs. How does this happen?

Analogous to the human brain, a neural network involves ‘neurons’ that can communicate with each other. These neurons are 
organized in several layers. The first layer is the input layer, the next are the ‘hidden layers’, followed by the output 
layer. Each neuron in a layer is actually a number. This number is called the ‘activation’ of that neuron. It determines the 
extent to which the pattern (e.g., visual, sound, etc.) picked up by that neuron weighs in on the output of that layer.

The activations in one layer are propagated forward to the next layer as a weighted sum. That is, the input of a neuron in one 
layer is the weighted sum of the outputs (i.e., activations) of the neurons in the previous layer. In addition, each neuron has 
a bias. The bias of a neuron can be thought of as its ‘threshold’ – its tendency to be on or off. This bias is similar to the 
intercept in a regression equation. In effect, the activations of the neurons in the preceding layer, the weights and the biases 
combine to determine the activation of a neuron. This activation value goes into an activation function, which converts it into 
the output of that neuron. I will use the sigmoid function for simplicity, which compresses the activation value into a number 
between zero and one.

The network reaches the correct response by learning from its mistakes. The quality of a response is communicated back to the 
network via back propagation. A loss function is defined, such as the sum of the squared differences. This calculates the 
distance between the correct response and the predicted response. The derivative of the loss function with respect to the 
weights gives the change in the loss function based on the change in the weights. The aim is to find the weights and the 
biases that minimize the loss function. This is done by an algorithm called the gradient descent. The gradient of a function 
is the direction in which it increases the most steeply. The negative of the gradient gives us the steepest decrease. The 
neural network minimizes the loss function by iterating according to the gradient descent. One can picture sliding on the 
slope of a function until we reach the lowest point on the curve.

## The Basics of the package 

### Required Python packages:
import numpy as np

### Required external modules and packages: 
from urllib.request import urlretrieve
import random
import os
import gzip

### The Data 

The Modified National Institute of Standards and Technology (MNIST) database is used to train and test the network 
(http://yann.lecun.com/exdb/mnist/). The data consist of 60,000 hand written digits and their corresponding labels in the 
'training' set and 10,000 digits and their labels in the 'test' set. Images are coded as 28x28 arrays. Each value in the 
array represents a pixel. 

### The Programme

The base code is Michael Nielsen's and can be found in http://neuralnetworksanddeeplearning.com. The following changes have 
been made to the code. 

**Loading the data**

The book has a separate programme to load and prepare the data. It creates a new validation data set from the original MNIST 
data. 10,000 images are taken from the training set to do this. In effect, the data used by the neural network has 50,000 
training examples, 10,000 test images, and 10,000 validation images. I did not use this programme and included different 
functions to prepare the data to be used in the neural network. 

I did this for two reasons. First, the validation sets are not used in the chapters with which this package is concerned. 
Specifically, this package is built upon the first two chapters. Therefore, I only used the original training and test data. 
The second reason I changed the data loading programme is that it has a lot of package dependencies. Some of these dependencies 
have changed over the years. As a result, using this programme was not straightforward to me. However, the functions I wrote 
draw from this programme and use the same data structures. Specifically, the data are stored in lists of tuples (x, y). For 
example, the training data has 60,000 of these tuples. x is a 28x28 = 784 dimensional array, each value representing a pixel. 
y is a 10 dimensional array. All values except one are 0s. The other value is 1. The element with value 1 is the digit 
between 0-9 which the network will classify.

**The backpropagation algorithm**

The book loops over the training examples in a mini-batch to compute the gradients. This package uses a matrix based approach adapted from Hindi Sellouk's code to speed up the process. In the original version of the code, a mini-batch is created by randomly sampling a certain number of training examples. The gradients are computed by propagating the activations forward and the error term backwards for each training example in the mini-batch, one by one. This process requires as many iterations as there are training examples in the mini-batch because the weight matrix is multiplied with the activation vector of each example in the mini-batch. 
Instead, the matrix approach stores all the activation vectors as the columns of a second matrix. This matrix is then multiplied with the weight matrix to feedforward all the training examples in the mini-batch simultaneously. Backpropagation works in a similar fashion. 

I have adapted the code to Python 3.x (it was coded in Python 2.x) and a few additions have been made to make testing easier. Also, I have tried to make the code easier on the eye.
