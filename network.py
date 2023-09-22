import numpy as np
import sys
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

class Network():
    def __init__(self, trainSet, trainLabels, testSet, testLabels, learnRate, epochs):
        self.layers = np.array([np.zeros(784), np.zeros(128), np.zeros(10)])
        self.weights = np.array([np.random.normal(0, np.sqrt(1/(794)), (784, 128)),
                       np.random.normal(0, np.sqrt(1/(794)), (128, 10))])
        self.adjustment = np.array([np.zeros((784, 128)), np.zeros((128, 10))])
        self.biases = np.array([np.zeros(128), np.zeros(10)])
         
        self.trainSet = trainSet
        self.trainLabels = trainLabels
        self.testSet = testSet
        self.testLabels = testLabels
        self.learnRate = learnRate
        self.epochs = epochs

    def sigmoid(self, x, derivative=0):
        if derivative:
            return (np.exp(-x))/((np.exp(-x) + 1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=0):
        exp = np.exp(x)
        if derivative:
            return 1
        return exp / np.sum(exp)

    def forwardStep(self):
        self.layers[1] = self.sigmoid(np.dot(self.weights[0].T, self.layers[0]))
        self.layers[2] = self.sigmoid(np.dot(self.weights[1].T, self.layers[1]))

    def computeAdjustment(self):
        commonLambda = (self.layers[2] - self.expected) * self.layers[2] * (1 - self.layers[2])

        self.adjustment[1] = np.transpose(np.outer(commonLambda, self.layers[1]))
        self.adjustment[0] = np.transpose(np.outer(np.dot(commonLambda, np.transpose(self.weights[1])) * 
                        self.layers[1] * (1 - self.layers[1]), self.layers[0]))
                        
    def backProp(self):
        for i in range(2):
            self.weights[i] -= self.learnRate * self.adjustment[i]
            #print('Adjustment ', i, ': ', self.adjustment[i], 'Weights ', i, ': ', self.weights[i], '\n\n\n')

    def computeAccuracy(self):
        guesses = []
        index = 0
        for sample in self.testSet:
            self.layers[0] = sample.flatten() / 255.0 * 0.99 + 0.01
            self.forwardStep()
            guess = np.argmax(self.layers[2])
            label = self.testLabels[index]
            guesses.append(guess == label)
            index += 1
        return np.mean(guesses) * 100

    def trainNetwork(self):
        for epoch in range(self.epochs):
            index = 0

            shuffle = np.random.permutation(len(self.trainSet))
            self.trainSet = self.trainSet[shuffle]
            self.trainLabels = self.trainLabels[shuffle]

            for sample in self.trainSet:
                print('Epoch ', epoch + 1, ': ', round(index / len(self.trainSet) * 100, 2), '% done', end='\r')

                self.expected = np.zeros(10) + 0.01
                self.layers[0] = sample.flatten() / 255.0 * 0.99 + 0.01
                self.expected[self.trainLabels[index]] = 0.99
                self.forwardStep()
                self.computeAdjustment()
                self.backProp()

                index += 1
            print('\nEpoch ', epoch + 1, ': ', self.computeAccuracy(), '% accuracy\n')

        print('\nTRAINING FINISHED\n')
        self.predict()

    def predict(self):
        index = int(input('Image index: '))
        while index >= 0:
            self.layers[0] = self.testSet[index].flatten() / 255.0 * 0.99 + 0.01
            self.forwardStep()

            plt.imshow(self.testSet[index])
            plt.title(np.argmax(self.layers[2]))
            plt.show()

            index = int(input('Image index: '))

