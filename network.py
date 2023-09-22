import numpy as np
import matplotlib.pyplot as plt

class Network():
    def __init__(self, trainSet, trainLabels, testSet, testLabels, learnRate, epochs):
        self.params = {
            'W1': np.random.normal(0, np.sqrt(1/(794)), (784, 128)),
            'W2': np.random.normal(0, np.sqrt(1/(794)), (128, 10)),
            'B1': np.random.normal(0, 1, 128),
            'B1': np.random.normal(0, 1, 10),
        } 
        self.trainSet = trainSet
        self.trainLabels = trainLabels
        self.testSet = testSet
        self.testLabels = testLabels
        self.learnRate = learnRate
        self.epochs = epochs

    def sigmoid(self, x, derivative=0):
        if derivative:
            return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=0):
        exp = np.exp(x - x.max())
        if derivative:
            return (exp / np.sum(exp)) * ((1 - exp) / np.sum(exp))
        return exp / np.sum(exp)

    def forwardStep(self):
        self.params['Z1'] = np.dot(self.params['Z0'].T, self.params['W1'])
        self.params['A1'] = self.sigmoid(self.params['Z1'])
        self.params['Z2'] = np.dot(self.params['A1'].T, self.params['W2'])
        self.params['A2'] = self.sigmoid(self.params['Z2'])

    def computeAdjustment(self):
        error = (self.params['A2'] - self.expected) * self.sigmoid(self.params['Z2'], derivative=1)
        self.params['dW2'] = np.outer(error, self.params['A1']).T
        
        error = np.dot(error, self.params['W2'].T) * self.sigmoid(self.params['Z1'], derivative=1)
        self.params['dW1'] = np.outer(error, self.params['Z0']).T
                        
    def backProp(self):
        self.params['W2'] -= self.learnRate * self.params['dW2']
        self.params['W1'] -= self.learnRate * self.params['dW1']

    def computeAccuracy(self):
        guesses = []
        index = 0
        for sample in self.testSet:
            self.params['Z0'] = sample.flatten() / 255.0 * 0.99 + 0.01
            self.forwardStep()
            guess = np.argmax(self.params['A2'])
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
                self.params['Z0'] = sample.flatten() / 255.0 * 0.99 + 0.01
                self.expected[self.trainLabels[index]] = 0.99
                self.forwardStep()
                self.computeAdjustment()
                self.backProp()

                index += 1
            print('\nEpoch ', epoch + 1, ': ', self.computeAccuracy(), '% accuracy\n')
            
            # Debug
            # print(self.params['dW1'], self.params['dW2'])
        
        print('\nTRAINING FINISHED\n')
        self.predict()

    def predict(self):
        index = int(input('Image index: '))
        while index >= 0:
            self.params['Z0'] = self.testSet[index].flatten() / 255.0 * 0.99 + 0.01
            self.forwardStep()

            plt.imshow(self.testSet[index])
            plt.title(np.argmax(self.params['A2']))
            plt.show()

            index = int(input('Image index: '))

