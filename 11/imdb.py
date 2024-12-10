import tensorflow as tf
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

class IMDB:
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
        '''Was having MemoryError issues, so I used SciPy's sparse matrix module to create a sparse matrix'''
        results = lil_matrix((len(sequences), dimension), dtype=bool)
        for i, sequence in enumerate(sequences):
            sequence = [min(idx, dimension-1) for idx in sequence]
            results[i, sequence] = 1  
        return results.astype(np.float32)  

    def prepare_data(self):
        '''Load the data and adjust format'''
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.imdb.load_data(NUM_WORDS=10000)

        self.x_train = self.vectorize_sequences(self.x_train)
        self.x_test = self.vectorize_sequences(self.x_test)

        self.y_train = np.asarray(self.y_train).astype('float32')
        self.y_test = np.asarray(self.y_test).astype('float32')

        self.x_val = self.x_train[:10000]
        self.y_val = self.y_train[:10000]

        self.x_train_partial = self.x_train[10000:]
        self.y_train_partial = self.y_train[10000:]   

    def build_model(self):
        '''Build the model'''
        self.model = tf.keras.Sequential([    
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self):
        '''Train the model'''
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2
            ),
            tf.keras.callbacks.TensorBoard()
        ]

        history = self.model.fit(self.x_train_partial, self.y_train_partial, epochs=20, batch_size=256,
                    validation_data=(self.x_val, self.y_val), callbacks=callbacks)
        
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

im = IMDB()
if __name__ == "__main__":
    print("IMDB class")
    im.prepare_data()
    im.build_model()
    im.train()
    im.plot_loss()
    im.plot_accuracy()
    prediction = im.evaluate()
    print(f"Loss and Accuracy: {prediction}")
