import mnist
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
import pickle

def convert(image_filename, size = (28,28)):
    '''Convert the image to grayscale, resize to 28x28, and conver to a numpy array'''
    new_image = Image.open(image_filename).convert('L') 
    new_image = new_image.resize(size) 
    new_image = np.array(new_image)
    new_image = 255 - new_image
    new_image = (new_image - np.min(new_image)) * (255 / (np.max(new_image) - np.min(new_image)))
    new_image = new_image.astype(np.float32) / 255.0  
    new_image = new_image.flatten()
    new_image = new_image.reshape((1, 784))
    return new_image

def predict(image_filename, converted_image, digit):
    '''Predicts which digit is shown by the image, and also displays the image to the screen'''
    network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

    with open(f'Aslam_mnist_model.pkl', 'rb') as f:
        network.params = pickle.load(f)
    
    network.update_layers()

    y_hat = network.predict(converted_image)
    p = np.argmax(y_hat)
    
    if digit == p:
        print(f'Success: {image_filename} is for digit {digit} is recognized as {p}.')
        plt.imshow(converted_image.reshape(28, 28))
        plt.show()
    else:
        print(f'Fail: Image {image_filename} is for digit {digit} but the inference result is {p}.')
        plt.imshow(converted_image.reshape(28, 28))
        plt.show()
    
if __name__ == "__main__":
    '''Running module6.py'''
    image_filename = sys.argv[1]
    digit = sys.argv[2]

    converted_image = convert(image_filename)

    predict(image_filename, converted_image, int(digit))