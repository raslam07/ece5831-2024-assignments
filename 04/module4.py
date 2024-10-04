import multilayer_percetron
import numpy as np

mlp = multilayer_percetron.MultiLayerPerceptron()

mlp.init_network()
y = mlp.forward(np.array([7.0, 2.0]))

print(f'The output is {y}')
