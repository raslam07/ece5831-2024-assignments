import numpy as np

class Errors:
    def cross_entropy_error(self, y, t):
        '''Returns the cross entropy error'''
        delta = 1e-7
        batch_size = 1 if y.ndim == 1 else y.shape[0]

        return -np.sum(t*np.log(y + delta)) / batch_size

if __name__ == "__main__":
    print("Errors Class")