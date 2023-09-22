from network import * 
import tensorflow as tf

(trainSet, trainLabels), (testSet, testLabels) = tf.keras.datasets.mnist.load_data()

# You can resize the training set 
# trainSet = np.resize(trainSet, (1000, 28, 28))
# trainLabels = np.resize(trainLabels, 1000)

# You can customize the learning rate and number of epochs
network = Network(trainSet, trainLabels, testSet, testLabels, 0.001, 10)
network.trainNetwork()