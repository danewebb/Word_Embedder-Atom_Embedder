import tensorflow as tf
import numpy as np
import os
# import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import pickle


batch_size = 20
set_epochs = 100
set_valsteps = 20
embedding_dim = 2
dense1_size = 16
dense2_size = 1



# clean, encoded training data
with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\train_vec.pkl', 'rb') as file:
    train_data = pickle.load(file)

# lengths of each atom
with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\train_len.pkl', 'rb') as file1:
    train_lens = pickle.load(file1)


    

# vocab in order of most common -> least common
with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\ranked_vocab.pkl', 'rb') as voc_file:
    vocab = pickle.load(voc_file)

vocab_size = len(vocab)






# non-csv path
# list -> ndarray
train_data_arr = np.array([np.array(x) for x in train_data])

# post padding
padded_train_data = keras.preprocessing.sequence.pad_sequences(
    train_data_arr, padding='post')

# dummy labels
tdata_size = np.shape(train_data_arr)
trlabels = np.zeros([tdata_size[0], 1])
trlabels = trlabels.astype('int32')

# convert train_vec into tf dataset. Could be valuable.

# td = tf.data.Dataset.from_tensors((train_data_arr, trlabels))
# td = tf.data.Dataset.from_tensors(train_data_arr)
# td = td.shuffle(1000, reshuffle_each_iteration = True)
#
# # td = td.shuffle(1000, reshuffle_each_iteration = True).padded_batch(10, padded_shapes=([None], ()))
# td = td.batch(batch_size, drop_remainder=True)



with tf.device('/cpu:0'): # gpu or cpu

    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(dense1_size, activation='relu'),
        layers.Dense(dense2_size)
    ])

    model.summary()


    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        padded_train_data, trlabels,
        epochs=set_epochs
        , batch_size=batch_size
        , shuffle=True
    )


# save model
keras.models.save_model(model, r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\ricocorpus_model')




# ### Retrieve Word Embeddings
# e = model.layers[0]
# weights = e.get_weights()[0]
# print(weights.shape)
# #
# import io
# #
# # # need to decode to get words rather than numbers
# # # encoder = info.features['text'].encoder
# #
# out_v = io.open(r'ricocorpus_wordembedding\vecs.tsv', 'w', encoding='utf-8')
# out_m = io.open(r'ricocorpus_wordembedding\meta.tsv', 'w', encoding='utf-8')
# #
# for num, word in enumerate(vocab):
#   vec = weights[num]
#   out_m.write(word + "\n")
#   out_v.write('\t'.join([str(x) for x in vec]) + "\n")
#
#
#
#
# out_v.close()
# out_m.close()




