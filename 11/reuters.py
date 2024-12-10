import tensorflow as tf
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

class Reuters:
    def __init__(self):
        '''Initialize the member variables'''
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.x_train_partial = None
        self.y_train_partial = None
        self.model = None
        self.history_dict = None
    
    def vectorize_sequences(self, sequences, dimension=10000):
        """Vectorize sequences with reduced memory usage"""
        results = np.zeros((len(sequences), dimension), dtype=np.float16)
        for i, sequence in enumerate(sequences):
            for j in sequence:
                if j < dimension:
                    results[i, j] = 1
        return results

    def prepare_data(self):
        '''Load the data and adjust format'''
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.reuters.load_data(num_words=10000)

        self.x_train = self.vectorize_sequences(self.x_train)
        self.x_test = self.vectorize_sequences(self.x_test)

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        num_train_samples = len(self.x_train)
        val_size = int(0.2 * num_train_samples)

        self.x_val = self.x_train[:val_size]
        self.y_val = self.y_train[:val_size]

        self.x_train_partial = self.x_train[val_size:]
        self.y_train_partial = self.y_train[val_size:]

    def build_model(self):
        '''Build the model'''
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(10000,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(46, activation="softmax")  
        ])

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        '''Train the model'''
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            tf.keras.callbacks.TensorBoard()
        ]

        history = self.model.fit(
            self.x_train_partial, self.y_train_partial,
            epochs=20, batch_size=256,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks
        )

        self.history_dict = history.history

    def plot_loss(self):
        '''Display the plot for loss'''
        loss_values = self.history_dict['loss']
        val_loss_values = self.history_dict['val_loss']
        epoches = range(1, len(loss_values) + 1)

        plt.plot(epoches, loss_values, 'b-.', label='Training Loss')
        plt.plot(epoches, val_loss_values, 'b-', label='Validation Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        '''Display the plot for accuracy'''
        accuracy_values = self.history_dict['accuracy']
        val_accuracy_values = self.history_dict['val_accuracy']
        epoches = range(1, len(accuracy_values) + 1)

        plt.plot(epoches, accuracy_values, 'g-.', label='Training Accuracy')
        plt.plot(epoches, val_accuracy_values, 'g-', label='Validation Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self):
        '''Return accuracy and loss on test set'''
        results = self.model.evaluate(self.x_test, self.y_test)
        return results

re = Reuters()
if __name__ == "__main__":
    print("Reuters class")
    re.prepare_data()
    re.build_model()
    re.train()
    re.plot_loss()
    re.plot_accuracy()
    prediction = re.evaluate()
    print(f"Loss and Accuracy: {prediction}")