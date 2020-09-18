import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # info and warning messages aren't printed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import re
import time
import pickle
# import tensorflow_datasets as tfds

import Process_sciart as PSA

# tfds.disable_progress_bar()


class Sciart_Word_Embedding():

    def __init__(self, data_chunks, model_path='', labels=0, vocab=0, vocab_size=0, batch_size=20, set_epochs=1, embedding_dim=2,
                 dense1_size=16, dense2_size=1):
        """

        :param data_chunks: List of data file paths. Break up data so memory doesn't overflow.
        :param model: First time through leave blank but once model is save in path, change zero to model path.
        :param vocab_size: If we are using scientific papers vocab built in Process_sciart, leave vocab_size=0
        :param labels: Dummy labels = 0, every label is a zero.
        :param batch_size:
        :param set_epochs:
        :param embedding_dim:
        :param dense1_size: Size of first Dense, 'relu', layer
        :param dense2_size: Size of second Dense 'default', layer
        """


        self.data_chunks = data_chunks
        if vocab_size == 0:
            # get vocab_size from 'sciart_vocab.pkl'
            self.vocab, self.vocab_size = self.voc()
        else:
            self.vocab = vocab
            self.vocab_size = len(self.vocab)

        self.labels = labels

        self.batch_size = batch_size
        self.set_epochs = set_epochs
        self.embedding_dim = embedding_dim
        self.dense1_size = dense1_size
        self.dense2_size = dense2_size


        if model_path == '':
            try:
                # saved model
                os.path.exists(r'C:\Users\liqui\PycharmProjects\THESIS\venv\Lib\sciart_model')
                self.model = keras.models.load_model(r'C:\Users\liqui\PycharmProjects\THESIS\venv\Lib\sciart_wordembedding\sciart_model')
            except:
                # build new model
                self.model = self.build_model()
        else:
            # load different model
            self.model = keras.models.load_model(model_path)

    def retrieve_word_embeddings(self, model):
        ### Retrieve Word Embeddings

        model = keras.models.load_model(model)

        e = model.layers[0]
        weights = e.get_weights()[0]
        print(weights.shape)
        #
        import io
        out_v = io.open(r'sciart_wordembedding\vecs.tsv', 'w', encoding='utf-8')
        out_m = io.open(r'sciart_wordembedding\meta.tsv', 'w', encoding='utf-8')
        #
        for num, word in enumerate(self.vocab[1:], start=1):
            vec = weights[num]
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")

        out_v.close()
        out_m.close()


    def data_to_numpy(self, data):
        # make encoded data into numpy arrays
        train_data_arr = np.array([np.array(x) for x in data])

        padded_train_data = keras.preprocessing.sequence.pad_sequences(
            train_data_arr, padding='post'
        )

        tdata_size = padded_train_data.shape

        return padded_train_data, tdata_size



    def build_model(self):
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.dense1_size, activation='relu'),
            layers.Dense(self.dense2_size)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    def train(self, trdata, trlabels):
        history = model.fit(
            trdata,
            trlabels,
            epochs=self.set_epochs,
            batch_size=self.batch_size,
            shuffle=True
        )

    def voc(self):
        with open(r'sciart_wordembedding\sciart_vocab.pkl', 'rb') as voc_file:
            vocab = pickle.load(voc_file)
        vocab_size = len(vocab)
        return vocab, vocab_size

    def main(self):

        for chunk in self.data_chunks:
            with open(chunk, 'rb') as chunk_file:
                data = pickle.load(chunk_file)
            trdata, tdata_size = self.data_to_numpy(data)
            if self.labels == 0:
                trlabels = np.zeros([tdata_size[0], 1])
                trlabels = trlabels.astype('int32')
            else:
                trlabels = self.labels
                
            self.train(trdata, trlabels)

        tf.keras.models.save_model(self.model, r'C:\Users\liqui\PycharmProjects\THESIS\venv\Lib\sciart_wordembedding\sciart_model')



if __name__ == '__main__':

    epochs = 60
    report_every = 5
    data_chunks = [r'sciart_wordembedding\sciart_data_01.pkl', r'sciart_wordembedding\sciart_data_02.pkl', r'sciart_wordembedding\sciart_data_03.pkl',
                   r'sciart_wordembedding\sciart_data_04.pkl', r'sciart_wordembedding\sciart_data_05.pkl', r'sciart_wordembedding\sciart_data_06.pkl',
                   r'sciart_wordembedding\sciart_data_07.pkl', r'sciart_wordembedding\sciart_data_08.pkl', r'sciart_wordembedding\sciart_data_09.pkl',
                   r'sciart_wordembedding\sciart_data_10.pkl', r'sciart_wordembedding\sciart_data_11.pkl', r'sciart_wordembedding\sciart_data_12.pkl']

    SWE = Sciart_Word_Embedding(data_chunks)

    start_time = time.time()
    for ii in range(1, epochs+1):
        SWE.main()
        if ii%report_every == 0:
            end_time = time.time()
            print(f'{ii}/{epochs} completed')

            elapsed_time = end_time - start_time
            reports_left = (epochs - ii)/report_every
            eta = elapsed_time*reports_left/3600 ### Hours
            print(f'Estimated time to completion: {eta} minutes')
            start_time = end_time


    # SWE.retrieve_word_embeddings(r'sciart_wordembedding\sciart_model')


