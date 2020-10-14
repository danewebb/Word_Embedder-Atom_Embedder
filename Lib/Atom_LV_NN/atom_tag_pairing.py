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
    # set_va = set(tuple(row) for row in va) # need to turn list of list into set of tuples otherwise error
    # set_vd = set(tuple(row) for row in vd)
    for tag_group in vd:
        # grabs the index of the tag group from the adjacency list
        # if tag_group in va:
        #     tagnum.append(va.index(set_tag))
        if tag_group != ['']:
            if any(tag_group.count(ele) > 1 for ele in tag_group):
                tag_group = lazy_tag_fix(tag_group)
            order_list = every_order(tag_group)
            if any(item in order_list for item in va):
                # if any of the orientations in order_list match tag group in va
                res = [ii for ii, val in enumerate(order_list) if val in va]
                if len(res) > 1:
                    ValueError('Multiple orientations of same tag group in va')

                resint = int(res[0])
                tagnum.append(va.index(order_list[resint]))
            else:
                print(f'Tag without va pair {tag_group}')
                tagnum.append(len(va))
        else:
            tagnum.append(len(va))

    lenva = len(va)
    lenvd = len(vd)
    lentagnum = len(tagnum)

    one_hot_labels = one_hotify(tagnum)
    labels =  one_hot_labels
    # naive way
    # tagvec = []
    # for num in tagnum:
    #     if num == lenva:
    #         tagvec.append(np.zeros([2, 1]))
    #     else:
    #         temp = proj[:, num]
    #         temp = temp.reshape((temp.shape[0], 1))
    #         tagvec.append(temp)
    #
    #
    # labels = np.concatenate(tagvec, axis=1)




    return labels

def one_hotify(tagnum):
    # assume undefined tag groups are defined as len(tagnum)
    rows = len(tagnum)
    cols =  max(tagnum)
    one_hot = np.zeros((rows, cols-1), dtype=np.int32) # -1 because we don't want throwaway values in one-hot
    for row, tag in enumerate(tagnum):
        if tag != cols:
            one_hot[row, tag] = 1

    return one_hot


def every_order(tag_group):
    len_group = len(tag_group)
    # orientations_num = math.factorial(len_group) # number of possible combinations
    poss_lists = list(itertools.permutations(tag_group, len_group))
    listify = []
    for jj in poss_lists:
        listify.append(list(jj))
    return listify

def lazy_tag_fix(tag_group):
    # some tag groups accidentally have doubled tags. This function removes duplicates
    res = []
    for tag in tag_group:
        if tag not in res:
            res.append(tag)
    return res




overall_labels = tag_pairing(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\train_data.json',
            r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\adjacency_train.json',
            r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\Projection.pkl')

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\train_labels.pkl', 'wb') as file:
    pickle.dump(overall_labels, file)