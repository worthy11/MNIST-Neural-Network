import numpy as np
import matplotlib.pyplot as plt

class Network():
    def __init__(self, trainSet, trainLabels, testSet, testLabels, learnRate = 0.01, batchSize = 1, epochs = 1):
        self.params = {
            'W1': np.random.normal(0, np.sqrt(1/(794)), (128, 784)),
            'W2': np.random.normal(0, np.sqrt(1/(794)), (10, 128)),
            'B1': np.zeros(128),
            'B2': np.zeros(10)
        }
        self.trainSet = np.resize(trainSet, (int(np.floor(len(trainSet) / batchSize)), batchSize, 784))
        self.trainLabels = np.resize(trainLabels, (int(np.floor(len(trainLabels) / batchSize)), batchSize))
        self.testSet = testSet
        self.testLabels = testLabels

        self.learnRate = learnRate
        self.batchSize = batchSize
        self.epochs = epochs

    def sigmoid(self, x, derivative=0):
        if derivative:
            return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=0):
        exp = np.exp(x - x.max())
        softmax = exp / np.sum(exp)
        if derivative:
            softmax = np.reshape(softmax, (1, -1))
            d_softmax = (softmax * np.identity(softmax.size) - softmax.transpose() @ softmax)
            return d_softmax
        return softmax
    
    def forwardStep(self):
        self.params['Z1'] = np.dot(self.params['W1'], self.params['Z0'].T).T + self.params['B1']
        self.params['A1'] = self.sigmoid(self.params['Z1'])
        self.params['Z2'] = (np.dot(self.params['A1'], self.params['W2'].T) + self.params['B2']).T
        self.params['A2'] = self.sigmoid(self.params['Z2'])

    def computeAdjustment(self):
        self.params['dW2'] = np.zeros((10, 128))
        self.params['dW1'] = np.zeros((128, 784))
        self.params['dB2'] = np.zeros(10)
        self.params['dB1'] = np.zeros(128)
        
        error2 = (self.params['A2'] - self.params['E'].T) * self.sigmoid(self.params['Z2'], derivative=1)
        error1 = np.dot(error2.T, self.params['W2']) * self.sigmoid(self.params['Z1'], derivative=1)
        if self.batchSize == 1:
            self.params['dW2'] = np.outer(error2, self.params['A1'])
            self.params['dW1'] = np.outer(error1, self.params['Z0'])
            self.params['dB2'] = error2
            self.params['dB1'] = error1
            
        else:
            for rowE2, rowE1, col2, col1 in zip(error2.T, error1, self.params['A1'], self.params['Z0']):
                self.params['dW2'] += np.outer(rowE2, col2)
                self.params['dW1'] += np.outer(rowE1, col1)
                self.params['dB2'] += rowE2
                self.params['dB1'] += rowE1


    def backwardStep(self):
        self.params['W2'] -= self.learnRate * self.params['dW2'] / self.batchSize
        self.params['W1'] -= self.learnRate * self.params['dW1'] / self.batchSize
        self.params['B2'] -= self.learnRate * self.params['dB2'] / self.batchSize
        self.params['B1'] -= self.learnRate * self.params['dB1'] / self.batchSize
                    
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
            shuffle = np.random.permutation(len(self.trainSet))
            self.trainSet = self.trainSet[shuffle]
            self.trainLabels = self.trainLabels[shuffle]

            for sample, label, index in zip(self.trainSet, self.trainLabels, range(len(self.trainSet))):
                print('Epoch', epoch + 1, ': Batch', index + 1, ',', round(index / len(self.trainSet) * 100, 2), '% done', end='\r')
                self.params['Z0'] = sample / 255.0 * 0.99 + 0.01
                self.params['E'] = np.zeros((self.batchSize, 10)) + 0.01
                if self.batchSize == 1:
                    self.params['Z0'] = np.squeeze(self.params['Z0'])
                    self.params['E'] = np.squeeze(self.params['E'])
                    self.params['E'][label[0]] = 0.99
                else:
                    for i in range(self.batchSize):
                        self.params['E'][i][label[i]] = 0.99
                self.forwardStep()
                self.computeAdjustment()
                self.backwardStep()
            print('\nEpoch', epoch + 1, ':', self.computeAccuracy(), '% accuracy\n')
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

