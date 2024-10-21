import mnist_data
import sys
import numpy as np
import matplotlib.pyplot as plt

mnist = mnist_data.MnistData()

def train_func(idx):
    '''Displays image and label from train set'''
    (train_images, train_labels), (_, _) = mnist.load()
    print(f'Label: {np.argmax(train_labels[idx])}')
    plt.imshow(train_images[idx].reshape(28,28))
    plt.show()

def test_func(idx):
    '''Displays image and label from test set'''
    (_, _), (test_images, test_labels) = mnist.load()
    print(f'Label: {np.argmax(test_labels[idx])}')
    plt.imshow(test_images[idx].reshape(28,28))
    plt.show()

if __name__ == "__main__":
    '''Takes in 2 inputs from the command line and outputs the specified image and label'''
    dataset = sys.argv[1]
    index = int(sys.argv[2])

    if dataset == 'train':
        train_func(index)
    elif dataset == 'test':
        test_func(index)
