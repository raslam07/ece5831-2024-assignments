# %% 
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import sys
import matplotlib.pyplot as plt

# %%
def load_image(file_path):
    '''Takes in an input (image path) from the command line and returns the image'''
    image = Image.open(file_path).convert("RGB")
    return image

# %%
def init():
    '''Initialization, does not return anything'''
    np.set_printoptions(suppress=True)

# %%
def load_my_model():
    '''Returns the model as well as the labels'''
    model = load_model("model/keras_model.h5")

    class_names = open("model/labels.txt", "r").readlines()

    return model, class_names

# %%
def prep_input(image):
    '''Returns the image after converting it to an array of a specified size'''
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data

# %%
def predict(model, class_names, data):
    '''Will output the class and confidence score'''
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

# %%
def display(image):
    '''Will output the picture to the screen'''
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# %%
if __name__ == "__main__":
    '''Takes in an input from the command line and will output class, confidence score, and a picture'''
    file_path = sys.argv[1]

    init()
    image = load_image(file_path)
    model, class_names = load_my_model()
    data = prep_input(image)
    predict(model, class_names, data)
    display(image)


# %%