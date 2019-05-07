import os
import numpy as np
import progressbar
import h5py as h5

uniq_words = 6385


def int2onehot(ngams, u_words):
    num_ngrams = ngams.shape[0]
    ngrams_onehot = np.zeros((num_ngrams, 2, u_words))

    bar = progressbar.ProgressBar()
    for i in bar(range(num_ngrams)):
        # print(i)
        idx0 = ngams[i][0]
        idx1 = ngams[i][1]
        ngrams_onehot[i][0][idx0] = 1
        ngrams_onehot[i][1][idx1] = 1
    return ngrams_onehot


if __name__ == '__main__':
    all_2grams = np.load("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams.npy")
    all_2grams_bad = np.load("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams_bad.npy")

    all_2grams_onehot = int2onehot(all_2grams, uniq_words)
    h5f = h5.File("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams_onehot.hdf5", "w")
    h5f.create_dataset("ngrams", data=all_2grams_onehot)
    h5f.close()

    # np.save("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams_onehot.npy", all_2grams_onehot)
    all_2grams_onehot = None

    all_2grams_onehot_bad = int2onehot(all_2grams_bad, uniq_words)
    h5f = h5.File("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams_onehot_bad.hdf5", "w")
    h5f.create_dataset("ngrams", data=all_2grams_onehot_bad)
    h5f.close()
