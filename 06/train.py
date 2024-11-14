import mnist
mnist_class = mnist.Mnist()

from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

import numpy as np
import pickle

(x_train, y_train), (x_test, y_test) = mnist_class.load()

network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

iterations = 10000
train_size = x_train.shape[0]
batch_size = 16
lr = 0.01

iter_per_ecoph = max(train_size/batch_size, 1)

train_losses = []
train_accs = []
test_accs = []

for i in range(iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)

    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= lr*grads[key]

    ## this is for plotting losses over time
    train_losses.append(network.loss(x_batch, y_batch))

    if i%iter_per_ecoph == 0:
        train_acc = network.accuracy(x_train, y_train)
        train_accs.append(train_acc)
        test_acc = network.accuracy(x_test, y_test)
        test_accs.append(test_acc)
        print(f'train acc, test_acc : {train_acc}, {test_acc}')

my_weight_pkl_file = 'Aslam_mnist_model.pkl'
with open(f'{my_weight_pkl_file}', 'wb') as f:
    print(f'Pickle: {my_weight_pkl_file} is being created.')
    pickle.dump(network.params, f)
    print('Done.')

if __name__ == "__main__":
    print("Training using two layer network with back propagation")