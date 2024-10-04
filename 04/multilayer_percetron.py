import numpy as np

class MultiLayerPerceptron:
    def __init__(self):
        self.net = {}
        pass

    def init_network(self):
        net = {}
        # Layer 1
        net['w1'] = np.array([[0.7, 0.9, 0.3], [0.5, 0.4, 0.1]])
        net['b1'] = np.array([1, 1, 1])
        #Layer 2
        net['w2'] = np.array([[0.2, 0.3], [0.4, 0.5], [0.22, 0.1234]])
        net['b2'] = np.array([0.5, 0.5])
        #Layer 3 <-- Output
        net['w3'] = np.array([[0.7, 0.1], [0.123, 0.314]])
        net['b3'] = np.array([0.1, 0.2])

        self.net = net
    
    def forward(self, x):
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.identity(a3)

        return y
    
    def identity(self, x):
        return x
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))