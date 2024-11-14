import numpy as np

class Activations:
    def sigmoid(self, x):
        '''Calculates using sigmoid equation'''
        return 1/(1 + np.exp(-x))
    
    # for multi-dimensional x
    def softmax(self, x):
        '''Calculates using softmax equation'''
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x)  
        return np.exp(x) / np.sum(np.exp(x))
    
if __name__ == "__main__":
    print("Activations Class")