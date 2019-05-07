import os
import numpy as np
import json
from models.embedding import W2Vec, w2v_gen
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn import decomposition
import torch
import h5py as h5
from progressbar import ProgressBar


def text2emb(txt_dict, model, h5file, mode="stack"):
    # emb_dct = {}
    bar = ProgressBar()
    for _class, data in bar(txt_dict.items()):
        # dct = {}
        _class_group = h5file.create_group(name=_class)
        for img, text in data.items():
            # img_group = _class_group.create_dataset(name=img)
            embedded_text = []
            for line in text:
                emb = model.predict(torch.tensor(line, dtype=torch.long))
                emb = emb.detach().numpy()
                if mode is "sum":
                    emb = np.sum(emb, axis=0)
                elif mode is 'mean':
                    emb = np.mean(emb, axis=0)
                elif mode is "stack":
                    emb = np.stack(emb)
                embedded_text.append(emb)
            embedded_text = np.stack(embedded_text, axis=0)
            _class_group[img] = embedded_text
            # img_group["text"] = embedded_text
            # dct.update({img: embedded_text})
        # emb_dct.update({_class: dct})
    h5file.close()


def get_vals(h5file):
    all_vals = []
    for k, v in h5file.items():
        for _, v1 in h5file[k].items():
            data = v1[:]
            for i in range(data.shape[0]):
                all_vals.append(data[i])
    return all_vals


if __name__ == '__main__':
    text_path = "/Users/anastasia/PycharmProjects/diploma/outputs/all_text2ints_no_freq_w.json"
    p2emb = "/Users/anastasia/PycharmProjects/diploma/outputs/history_no_f_w_from250_1/model_904_199"

    with open(text_path, "r") as fp:
        text = json.load(fp)

    net = W2Vec(6376, 100, p2emb)

    h5file = h5.File("/Users/anastasia/PycharmProjects/diploma/outputs/all_text2emb_no_freq_w_stack.hdf5", "w")
    text2emb(text, net, h5file, mode="sum")

    # p2group = "/Users/anastasia/PycharmProjects/diploma/outputs/all_text2emb_sum.hdf5"
    # f = h5.File(p2group, "r")
    # #
    # all_embedding = get_vals(f)
    # # print("ok")
    # #
    # all_embedding = np.stack(all_embedding)
    # #
    # pca = decomposition.PCA(n_components=3)
    # #
    # pca.fit(all_embedding)
    # emb_3d = pca.transfrom(all_embedding)

    # all_embedding = np.load("/Users/anastasia/PycharmProjects/diploma/outputs/all_text2emb_mean_pca_3d.npy")
    # cluster = DBSCAN(eps=0.1, min_samples=40, metric="cosine", n_jobs=-1)
    # cluster.fit(all_embedding)
    # labels = cluster.labels_
    # print(labels)
