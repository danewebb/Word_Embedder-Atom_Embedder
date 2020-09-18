import tensorflow as tf
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import pickle


# def padding(data, data_len, maxlen=0):
#     old_num = 0
#     if maxlen == 0:
#         # find necessary padding length
#         for num in data_len:
#             if num > old_num:
#                 old_num = num
#
#         padlen = old_num
#     else:
#         padlen = maxlen
#
#     for arr in data:
#         while len(arr) < padlen:
#             arr.append(0)
#
#     return data

# def arrange_vocab(vocab):
#     lst = len(vocab)
#     arr_vocab = list()
#     for ii in range(0, lst-1):
#         for key, value in vocab.items():
#             if value == ii:
#                 arr_vocab.append(key)
#
#     return arr_vocab




batch_size = 20
set_epochs = 1000
set_valsteps = 20
embedding_dim = 16
dense1_size = 16
dense2_size = 1

with open('train_vec.pkl', 'rb') as file:
    train_data = pickle.load(file)

with open('train_len.pkl', 'rb') as file1:
    train_lens = pickle.load(file1)


    
# train_data_arr = np.array([np.array(x) for x in train_data])

with open('ranked_vocab.pkl', 'rb') as voc_file:
    vocab = pickle.load(voc_file)

vocab_size = len(vocab)

# vocab = arrange_vocab(vocab)

# padded_train_data = padding(train_data, train_lens)




# csv path

# with open('train_data.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(padded_train_data)
#
# train_csv_data = tf.data.experimental.make_csv_dataset(
#     padded_train_data,
#     batch_size,
#     num_epochs=set_epochs
# )


# non-csv path
# train_data_arr = np.array([np.array(x) for x in padded_train_data])
train_data_arr = np.array([np.array(x) for x in train_data])


padded_train_data = keras.preprocessing.sequence.pad_sequences(
    train_data_arr, padding='post')


tdata_size = np.shape(train_data_arr)

trlabels = np.zeros([tdata_size[0], 1])
trlabels = trlabels.astype('int32')
#convert train_vec into tf

# td = tf.data.Dataset.from_tensors((train_data_arr, trlabels))
# td = tf.data.Dataset.from_tensors(train_data_arr)
# td = td.shuffle(1000, reshuffle_each_iteration = True)
#
# # td = td.shuffle(1000, reshuffle_each_iteration = True).padded_batch(10, padded_shapes=([None], ()))
# td = td.batch(batch_size, drop_remainder=True)





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




### Retrieve Word Embeddings
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
#
import io
#
# # need to decode to get words rather than numbers
# # encoder = info.features['text'].encoder
#
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#
for num, word in enumerate(vocab):
  vec = weights[num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")




out_v.close()
out_m.close()




