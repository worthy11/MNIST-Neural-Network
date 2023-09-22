# MNIST Neural Network
A simple neural network I created to learn the mathematics behind backpropagation. The network trains on the MNIST dataset and learns to recognize handwritten digits.


## How it works
This neural network is composed of an input layer, an output layer and one hidden layer with 128 nodes. Weights are initialized according to Normal Xavier Initialization and the sigmoid and softmax are used as the activation functions. As the network trains on the 60 000 (unless specified otherwise) labeled samples available in the MNIST training dataset, you are informed about the % completeness of each epoch. The accuracy of the network is computed by making predictions on the 10 000 samples from the testing MNIST dataset.


## Customizing the network
The network can be customized through modifying the learning rate and numeber of epochs, as well as resizing the training dataset. The specific lines to be altered are appropriately labeled in the `main.py` file.