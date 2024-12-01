import pathlib
import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

class dogs_cats:
    def __init__(self):
        '''Initialize the datasets and the model'''
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.model = None
        self.base_dir = pathlib.Path('dogs-vs-cats')
        self.src_dir = pathlib.Path('dogs-vs-cats-original/train')

    def make_dataset_folders(self, subset_name, start_index, end_index):
        '''Make folders for each dataset'''
        for category in ("dog", "cat"):
            dir = self.base_dir / subset_name / category
            #print(dir)
            if os.path.exists(dir) is False:
                os.makedirs(dir)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            #print(files)
            for i, file in enumerate(files):
                shutil.copyfile(src=self.src_dir / file, dst=dir / file)
                if i % 100 == 0: # show only once every 100
                    print(f'src:{self.src_dir / file} => dst:{dir / file}')
    
    def _make_dataset(self, subset_name):
        '''Make dataset based on the passed in subset name'''
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.base_dir / subset_name,
            image_size=(180,180),
            batch_size=32
        )

        return dataset
    
    def make_dataset(self):
        '''Make dataset for train, validation, and test sets'''
        self.train_dataset = self._make_dataset('train')
        self.validation_dataset = self._make_dataset('validation')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
        '''Build the model'''
        inputs = layers.Input(shape=(180, 180, 3))
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.AveragePooling2D(pool_size=2)(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        
        return self.model
    
    def train(self, model_name):
        '''Perform training of the model and display its plot'''
        model_name = self.build_network()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.keras'),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]

        history = model_name.fit(self.train_dataset, epochs=20, 
                            validation_data=self.validation_dataset, 
                            callbacks=callbacks)
        
        acc = history.history['accuracy']
        loss = history.history['loss']

        plt.plot(range(1, len(acc)+1), acc, label='Traiing Acc')
        plt.show()

    def load_model(self, model_name):
        '''Load the model'''
        self.model = load_model(f"{model_name}.keras")
        print("Successfully loaded model")
    
    def predict(self, image_file):
        '''Make prediction on a single image and display the image'''
        image = Image.open(image_file)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
        image = tf.convert_to_tensor(image, dtype=tf.float32)  
        image = tf.image.resize(image, (180, 180)) 
        image = image / 255.0  
        image = tf.expand_dims(image, axis=0)  
        image = tf.tile(image, [32, 1, 1, 1]) 

        predictions = self.model.predict(image)

        return predictions

dg = dogs_cats()
if __name__ == "__main__":
    '''Trains the model and makes prediction on the cat.1500.jpg image'''
    print("dogs_cats class")
    dg.make_dataset()
    dg.train('Aslam_model')
    prediction = dg.predict("cat.1500.jpg")
    print(prediction)