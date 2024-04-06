# MNIST Neural Network
A simple neural network I created to learn the mathematics behind backpropagation. The network trains on the MNIST dataset and learns to recognize handwritten digits: <br />

<p align="center">
  <img src="https://github.com/worthy11/MNIST-Neural-Network/blob/main/mnist.png" alt="Sample from MNIST dataset"/>
</p>

## How it works
This neural network is composed of an input layer, an output layer and one hidden layer with 128 nodes. Weights are initialized according to Normal Xavier Initialization and the sigmoid and softmax are used as the activation functions. The network trains on 60 000 (unless specified otherwise) labeled data points available in the MNIST training dataset, providing feedback about the current batch # and completeness of each epoch. The accuracy of the network is computed by making predictions on 10 000 unseen data points from the testing MNIST dataset. After training is complete, you can select any of the 10 000 unseen data points and test whether or not the network labels it correctly. <br />

Implementation was achieved from scratch in Numpy, as well as using Torch.

## Customizing the network
The network can be customized through modifying the learning rate, batch size and number of epochs, as well as resizing the training dataset. The specific variables to be altered are appropriately labeled in the `main.py` file.
