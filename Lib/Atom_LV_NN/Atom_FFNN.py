import os
import pickle
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Atom_FFNN:
    def __init__(self, data_path, batch_size=20, epochs=100, model_path = '', training=False,
                 test_path='', train_label_path='', test_label_path='',
                 dense1=1172, dense2=2):
        try:
            with open(data_path, 'rb') as data_file:
                self.data = pickle.load(data_file)
        except ValueError:
            print('Data path does not exist')

        if not isinstance(self.data, np.ndarray):
            self.data = self.data_to_numpy(self.data)
        else:
            TypeError('data should be either ndarray or list')

        self.batch_size = batch_size
        self.epochs = epochs

        self.dense1 = dense1
        self.dense2 = dense2




        if model_path != '' and os.path.exists(model_path):
            self.model = keras.model.load_model(model_path)
        else:
            print('Building new model')
            self.model = self.build_model()

        if training: # True
            if test_path != '':
                if os.path.exists(test_path):
                    with open(test_path, 'rb') as test_file:
                        self.test_data = pickle.load(test_file)
                    if not isinstance(self.test_data, np.ndarray):
                        self.test_data = self.data_to_numpy(self.test_data)
                    else:
                        TypeError('test_data should be either ndarray or list')

                    with open(test_label_path, 'rb') as test_label_file:
                        self.test_labels = pickle.load(test_label_file)
                    if not isinstance(self.test_labels, np.ndarray):
                        self.test_labels = self.data_to_numpy(self.test_labels)
                    else:
                        TypeError('test_labels should be either ndarray or list')
                else:
                    raise Warning('Either test data or test labels are not defined. Please provide a valid path.')

            if os.path.exists(train_label_path):
                with open(train_label_path, 'rb') as train_label_file:
                    self.train_labels = pickle.load(train_label_file)
                if not isinstance(self.train_labels, (np.ndarray)):
                    self.train_labels = self.data_to_numpy(self.train_labels)
                else:
                    TypeError('train_labels should be either ndarray or list')
            else:
                raise Exception('Train label path is not valid.')



    def data_to_numpy(self, lst):
        # make encoded data into numpy arrays
        holder = []
        for ele in lst:
            # converts list of lists into a numpy array
            if ele == []:
                # check if empty list, not sure why empty lists are in the data.
                ele = [0., 0.]
            temp = np.array(ele)
            temp = temp.reshape((temp.shape[0], 1))
            holder.append(temp)

        arr = np.concatenate(holder, axis=1)


        return arr



    def padding(self, data):
        # array padding with zeros
        padded_data = keras.preprocessing.sequence.pad_sequences(
            data, padding='post'
        )
        data_size = padded_data.shape

        return padded_data, data_size

    def build_model(self):
        model = keras.Sequential([
            # layers.Dense(self.dense3, activation='relu'),
            # layers.GlobalAveragePooling1D(),

            layers.Dense(self.dense2, activation='relu', input_shape=(1, 2)),
            layers.Dense(self.dense1)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    def train(self):
        try:
            history = self.model.fit(
                self.data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True # shuffles batches
                ,verbose=1 # supresses progress bar
            )
        except ValueError:
            self.data = self.data.T
            history = self.model.fit(
                self.data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True
                # shuffles batches
                , verbose=1  # supresses progress bar
            )

        return history

    def save_model(self):
        keras.models.save_model(
            self.model, r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\atom_FFNN_model'
        )






if __name__ == '__main__':
    set_batch_size = 10
    set_epochs = 100
    AFF = Atom_FFNN(
        r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\atom_vectorsLV1.pkl', # local atom vector path
        batch_size=set_batch_size,
        epochs=set_epochs,
        training=True,
        train_label_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\train_labels.pkl' # local training label path
    )


    AFF.train()
