import os
import numpy as np
import pickle
import json
import re
import math
import itertools

def tag_pairing(data_dict, adjacency, projection):
    with open(data_dict, 'rb') as f1:
        data = json.load(f1)

    with open(adjacency, 'rb') as f2:
        adj = json.load(f2)

    with open(projection, 'rb') as f3:
        proj = pickle.load(f3)

    pattern = r'\w+'

    va = [] # adjacency values
    vd = [] # data values
    for v1 in data.values():
        val1 = v1['tag_dict']
        vd.append(val1['tags'])

    for val2 in adj['nodes'].keys():
        strip = re.findall(pattern, val2)
        va.append(strip)



    # grab index of matching tag intersection in va for each vd tag
    # data_tag_idxs = [idx for idx, tag in enumerate(vd) if tag in set(va)]

    tagnum = [] # stores adjacency tag groups.
    set_va = set(tuple(row) for row in va) # need to turn list of list into set of tuples otherwise error
    for tag_group in vd:

        # grabs the index of the tag group from the adjacency list
        if tag_group in va:
            tagnum.append(va.index(tag_group))
        # else:
            # order_list = every_order(tag_group)
            # if any(item in order_list for item in va):
            #     # if any of the orientations in order_list match tag group in va
            #     res = [ii for ii, val in enumerate(order_list) if val in set_va]
            #     if len(res) > 1:
            #         print(f'More than 1 orientation for tag_group ({tag_group})')
            #     else:
            #         print('function every order is necessary')
            #         tagnum.append(res)
        else:
            tagnum.append(len(va))

    lenva = len(va)
    lenvd = len(vd)
    lentagnum = len(tagnum)


    # naive way
    tagvec = []
    for num in tagnum:
        if num == lenva:
            tagvec.append(np.zeros([2, 1]))
        else:
            temp = proj[:, num]
            temp = temp.reshape((temp.shape[0], 1))
            tagvec.append(temp)


    labels = np.concatenate(tagvec, axis=1)

    # one_hot_DTI = np.zeros((len(vd), len(va)+1)) # +1 to have a null value
    #
    # one_hot_DTI[np.arange(len(tagnum)), tagnum] = 1
    # one_hot = np.delete(one_hot_DTI, len(va), 1)
    #
    #
    # for ii in range(0, len(va)):
    #     labels[ii][:] = proj[]*one_hot





    return labels

# def every_order(tag_group):
# not necessary
#     len_group = len(tag_group)
#     # orientations_num = math.factorial(len_group) # number of possible combinations
#     poss_lists = list(itertools.permutations(tag_group, len_group))
#
#
#     return poss_lists






overall_labels = tag_pairing(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\train_data.json',
            r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\adjacency_train.json',
            r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\Projection.pkl')

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\train_labels.pkl', 'wb') as file:
    pickle.dump(overall_labels, file)