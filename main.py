from nodeep import * 
import tensorflow as tf

(trainSet, trainLabels), (testSet, testLabels) = tf.keras.datasets.mnist.load_data()
trainSet = np.resize(trainSet, (100, 28, 28))
trainLabels = np.resize(trainLabels, 100)
network = Network(trainSet, trainLabels, testSet, testLabels, 0.1, 100)
network.trainNetwork()