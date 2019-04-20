import os
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, cdist

def find_closest(dist_arr, const=100):
    np.fill_diagonal(dist_arr, const)
    dct = {}
    for i in range(dist_arr.shape[0]):
        idx = np.argmin(dist_arr[i])
        #dist_arr[i] = const
        dct.update({i:idx})
    return dct


if __name__ == '__main__':
    # with open("/Users/anastasia/PycharmProjects/diploma/outputs/corpus_int2word.json", "r") as fp:
    #     int2w = json.load(fp)
    with open("/Users/anastasia/PycharmProjects/diploma/outputs/corpus_word2int.json", "r") as fp:
        w2int = json.load(fp)

    int2w = {v:k for k,v in w2int.items()}
    p2emb = "/Users/anastasia/PycharmProjects/diploma/outputs/model_804_50"
    embedding = torch.load(p2emb)

    voc_size = 6385

    all_inp = np.asarray(list(range(voc_size)))

    all_inp = torch.tensor(all_inp, dtype=torch.long)
    out = embedding(all_inp)
    out = out.detach().numpy()

    dist = cdist(out, out, 'cosine')
    distance_dct = find_closest(dist)
    distance_dct_words = {int2w[k]:int2w[v] for k,v in distance_dct.items()}
    print("ok")

    with open("/Users/anastasia/PycharmProjects/diploma/outputs/distance_dict.json", "w") as fp:
        json.dump(distance_dct_words, fp)
    # with open("/Users/anastasia/PycharmProjects/diploma/outputs/corpus_int2word.json", "w") as fp:
    #     json.dump(int2w, fp)

