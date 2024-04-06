import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

class Network():
    def __init__(self, trainSet, testSet, learnRate = 0.01, batchSize = 1, epochs = 1):
        with torch.no_grad():
            self.params = {
                'W1': torch.empty(128, 784, requires_grad=True).normal_(mean=0, std=np.sqrt(1/(794))),
                'W2': torch.empty(10, 128, requires_grad=True).normal_(mean=0, std=np.sqrt(1/(794))),
                'B1': torch.zeros(128, requires_grad=True),
                'B2': torch.zeros(10, requires_grad=True)
            }
        self.trainSet = trainSet
        self.testSet = testSet

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD([self.params['W1'], self.params['W2'], self.params['B1'], self.params['B2']], lr=learnRate)
        self.learnRate = learnRate
        self.batchSize = batchSize
        self.epochs = epochs

    def sigmoid(self, x, derivative=0):
        if derivative:
            return torch.exp(-x) / ((torch.exp(-x) + 1) ** 2)
        return 1/(1 + torch.exp(-x))
    
    def forwardStep(self):
        self.params['Z1'] = (torch.matmul(self.params['W1'], self.params['Z0'].T) + self.params['B1'][:, None]).T
        self.params['A1'] = self.sigmoid(self.params['Z1'])
        self.params['Z2'] = (torch.matmul(self.params['A1'], self.params['W2'].T).T + self.params['B2'][:, None]).T
        self.params['A2'] = self.sigmoid(self.params['Z2'])
                    
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
            for idx, (image, label) in self.trainSet:
                print('Epoch', epoch + 1, ': Batch', idx + 1, ',', round(idx / (60000/self.batchSize) * 100, 2), '% done', end='\r')
               
                self.params['Z0'] = image.resize(self.batchSize, 784) / 255.0 * 0.99 + 0.01
                self.params['E'] = np.zeros((self.batchSize, 10)) + 0.01
                for i in range(self.batchSize):
                    self.params['E'][i][label[i]] = 0.99

                self.forwardStep()
                loss = torch.mean((self.params['E'], self.params['A2'])**2 / 10)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

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

