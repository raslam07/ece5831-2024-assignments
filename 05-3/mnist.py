import mnist_data
import numpy as np
import pickle

class Mnist():
    def __init__(self):
        '''Initialize the data and parameters'''
        self.data = mnist_data.MnistData()
        self.params = {}


    def sigmoid(self, x):
        '''Sigmoid function implementation'''
        return 1/(1 + np.exp(-x))


    def softmax(self, a):
        '''Softmax function implementation'''
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a/np.sum(exp_a)
    

    def load(self):
        '''Return the training and testing data'''
        (x_train, y_train), (x_test, y_test) = self.data.load()
        return (x_train, y_train), (x_test, y_test)
    
    
    def init_network(self):
        '''Initialize the network'''
        with open('model/sample_weight.pkl', 'rb') as f:
            self.params = pickle.load(f)
    

    def predict(self, x):
        '''Make prediction using neural network weights and biases'''
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.softmax(a3)

        return y    