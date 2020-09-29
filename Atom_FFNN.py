import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Atom_FFNN:
    def __init__(self, data_path, batch_size, epochs, model_path = '', training=False, test_path='', train_label_path='',
                 test_label_path='', dense1=2, dense2=32, dense3=64):
        try:
            with open(data_path, 'rb') as data_file:
                self.data = pickle.load(data_file)
        except ValueError:
            print('Data path does not exist')

        self.batch_size = batch_size
        self.epochs = epochs

        if model_path != '' and os.path.exists(model_path):
            self.model = keras.model.load_model(model_path)
        else:
            self.model = self.build_model

        if training: # True
            if test_path != '':
                if os.path.exists(test_path):
                    with open(test_path, 'rb') as test_file:
                        self.test_data = pickle.load(test_file)

                    with open(test_label_path, 'rb') as test_label_file:
                        self.test_labels = pickle.load(test_label_path)

                else:
                    raise Warning('Either test data or test labels are not defined. Please provide a valid path.')

            if os.path.exists(train_label_path):
                with open(train_label_path, 'rb') as train_label_file:
                    self.train_labels = pickle.load(train_label_file)

            else:
                raise Exception('Train label path is not valid.')



        self.dense1 = dense1
        self.dense2 = dense2
        self.dense3 = dense3

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(self.dense3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.dense2, activation='relu'),
            layers.Dense(self.dense1)
        ])

        return model

    def train(self, data, labels):
        history = self.model.fit(
            data, labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True # shuffles batches
            ,verbose=0 # supresses progress bar
        )

        return history






