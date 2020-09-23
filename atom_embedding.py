import numpy as np
import os
import pickle
import tensorflow as tf




class Atom_Embedder():

    def __init__(self, weights, vocab):
        # require each paragraph to be fed in to class?

        # self.atoms = listof_atoms # discrete unit of text, e.g. chapters, paragraphs, sentences.
        self.weights = weights # model weights from word embeddings
        self.vocab = vocab


        # dictionary of word encoded values as keys, and the vectors as values
        self.encoded_vecs = dict()
        self.__assign_vecs_to_enc()

    def __assign_vecs_to_enc(self):
        # Assigns corresponding vectors to the encoded word values.
        for num, word in enumerate(self.vocab):  # 0 index of vocab is 0????
            # loop through vocab and assign the correct weights to the vocab
            self.encoded_vecs[num] = self.weights[num]




    def sum_of_difference(self, atom):

        #### Need to rewrite this to work with vector addition/subtraction.

        diffs = []
        for ii in range(0, len(atom) - 1): # -1 or -2????
            diffs.append(self.encoded_vecs[atom[ii+1]] - self.encoded_vecs[atom[ii]])

        res = [sum(jj) for jj in zip(*diffs)]

        return res



    def sum_atoms(self, atom):
        sums = []
        for ii in range(0, len(atom) - 1): # 0  or -1 ???
            sums.append(self.encoded_vecs[atom[ii]])

        res = [sum(jj) for jj in zip(*sums)]

        return res



if __name__ == '__main__':
    atom_vecs = []
    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\ranked_vocab.pkl', 'rb') as voc_file:
        vocab = pickle.load(voc_file)

    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\train_vec.pkl', 'rb') as vec_file:
        train_vectors = pickle.load(vec_file)

    model = tf.keras.models.load_model(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\ricocorpus_model')

    AE = Atom_Embedder(model.layers[0].get_weights()[0], vocab)

    for para in train_vectors:
        # atom_vecs.append(AE.sum_of_difference(para))
        atom_vecs.append(AE.sum_atoms(para))
    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\atom_vectors', 'wb') as file:
        pickle.dump(atom_vecs, file)




