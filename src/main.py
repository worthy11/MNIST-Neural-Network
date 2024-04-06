from model_np import * 
import tensorflow as tf

(trainSet, trainLabels), (testSet, testLabels) = tf.keras.datasets.mnist.load_data()

# --- RESIZE THE TRAINING SET HERE ---
# trainSet = np.resize(trainSet, (1000, 28, 28))
# trainLabels = np.resize(trainLabels, 1000)

# --- CUSTOMIZE THE LEARNING RATE, BATCH SIZE AND NUMBER OF EPOCHS HERE ---
network = Network(trainSet, trainLabels, testSet, testLabels, learnRate=0.1, batchSize=16, epochs=5)
network.trainNetwork()