# Neural Networks 

**'sensus' is the name of my package. It means 'perception' or 'sense' in Latin. It was the only name I could 
come up with that was both relatable to neural networks and was unique among Python packages. The code is in the package folder.**

*In this project, I want to build a neural network that can recognize handwritten digits, using Python. The functions involved in a simple neural network are clear and the training data are available online. For the functions to make sense, I wrote a brief description of how a neural network works. I should be able to finish this on time and I do not know what I can cut if I run out of time. If anything is left out from the structure I explained below, it will not work. If I have time left, I will build a Shiny app that shows the effect of drift in Brownian motion simulations.*

A neural network is a computing system that can learn from data. First, training data are provided to the neural network. Next, these are converted into an output (e.g., producing the output 7 in response to a handwritten 4). Then, the neural network receives feedback about its performance. This feedback is given by communicating the discrepancy between its decision and the correct answer back to the network. The network minimizes this discrepancy by adjusting its outputs. How does this happen? 

Analogous to the human brain, a neural network involves ‘neurons’ that can communicate with each other. These neurons are organized in several layers. The first layer is the input layer, the next are the ‘hidden layers’, followed by the output layer. Each neuron in a layer is actually a number. This number is called the ‘activation’ of that neuron. It determines the extent to which the pattern (e.g., visual, sound, etc.) picked up by that neuron weighs in on the output of that layer. 

The activations in one layer are propagated forward to the next layer as a weighted sum. That is, the input of a neuron in one layer is the weighted sum of the outputs (i.e., activations) of the neurons in the previous layer. In addition, each neuron has a bias. The bias of a neuron can be thought of as its ‘threshold’ – its tendency to be on or off. This bias is similar to the intercept in a regression equation. In effect, the activations of the neurons in the preceding layer, the weights and the biases combine to determine the activation of a neuron. This activation value goes into an activation function, which converts it into the output of that neuron. I will use the sigmoid function for simplicity, which compresses the activation value into a number between zero and one. 

The network reaches the correct response by learning from its mistakes. The quality of a response is communicated back to the network via back propagation. A loss function is defined, such as the sum of the squared differences. This calculates the distance between the correct response and the predicted response. The derivative of the loss function with respect to the weights gives the change in the loss function based on the change in the weights. The aim is to find the weights and the biases that minimize the loss function. This is done by an algorithm called the gradient descent. The gradient of a function is the direction in which it increases the most steeply. The negative of the gradient gives us the steepest decrease. The neural network minimizes the loss function by iterating according to the gradient descent. One can picture sliding on the slope of a function until we reach the lowest point on the curve. 

To summarize, these are the functions I will use in this project. 
1. The activation function defines the nodes and their outputs
2. The weights and the biases determine the inputs to subsequent layers 
3. The loss function is the feedback mechanism
4. Gradient descent optimizes learning from the feedback  
