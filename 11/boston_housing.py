import tensorflow as tf
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

class BostonHousing:
    def __init__(self):
        '''Initialize the member variables'''
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.all_mae_histories = []

    def prepare_data(self):
        '''Load the data and adjust format'''
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.boston_housing.load_data()

        mean = self.x_train.mean(axis=0)
        std = self.x_train.std(axis=0)
        self.x_train = (self.x_train - mean) / std
        self.x_test = (self.x_test - mean) / std

    def build_model(self):
        '''Build the model'''
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

        return model

    def train(self):
        '''Train the model using k-fold validation'''
        k = 4  
        num_val_samples = len(self.x_train) // k
        num_epochs = 100 

        self.all_mae_histories = [] 

        for i in range(k):
            print(f"Processing fold #{i}")
            
            val_data = self.x_train[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.y_train[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [self.x_train[:i * num_val_samples], self.x_train[(i + 1) * num_val_samples:]],
                axis=0
            )
            partial_train_targets = np.concatenate(
                [self.y_train[:i * num_val_samples], self.y_train[(i + 1) * num_val_samples:]],
                axis=0
            )

            self.model = self.build_model()
            history = self.model.fit(
                partial_train_data, partial_train_targets,
                validation_data=(val_data, val_targets),
                epochs=num_epochs, batch_size=16, verbose=0
            )

            mae_history = history.history["val_mae"]
            self.all_mae_histories.append(mae_history)

    def plot_loss(self):
        '''Display the plot for loss'''
        num_epochs = len(self.all_mae_histories[0])
        average_mae_history = [np.mean([x[i] for x in self.all_mae_histories]) for i in range(num_epochs)]

        plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
        plt.title("Validation MAE Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.grid(True)
        plt.show()

    def evaluate(self):
        '''Return accuracy and MAE on test set'''
        results = self.model.evaluate(self.x_test, self.y_test)
        return results

bh = BostonHousing()
if __name__ == "__main__":
    print("Boston Housing Class")
    bh.prepare_data()
    bh.build_model()
    bh.train()
    bh.plot_loss()
    prediction = bh.evaluate()
    print(f"Loss and MAE: {prediction}")