import os
import numpy as np
import json
from itertools import chain
#from progressbar import ProgressBar
from multiprocessing import Pool

seed = 896723765
#uniq_words = 6385


uniq_words_no_freq = 6376

pairs_dict = None


def line2ngrams(line, n=2):
    ln = len(line)
    ngrams = [[line[i+j] for j in range(n)] for i in range(ln-(n-1))]
    return ngrams


def anti_ngram(n_gram):
    a, b = n_gram
    try:
        a_neighbors = pairs_dict[a]
    except KeyError:
        a_neighbors = []
    try:
        b_neighbors = pairs_dict[b]
    except KeyError:
        b_neighbors = []
    neighbors = list(set(a_neighbors + b_neighbors))

    if neighbors == []:
        return []
    all_words = list(range(uniq_words_no_freq))
    for n in neighbors:
        all_words.remove(n)
    if all_words == []:
        return []
    else:
        np.random.seed(seed)
        anti_b = np.random.choice(all_words)
        return [a, anti_b]


if __name__ == '__main__':
    p2txt = "/Users/anastasia/PycharmProjects/diploma/outputs/all_text2ints_no_freq_w.json"

    with open(p2txt, "r") as fp:
        text = json.load(fp)

    text2list = list(chain.from_iterable([list(v) for class_id in text.values() for v in class_id.values()]))
    pairs_dct = {}
    all_ngrams = []

    for line in text2list:
        ngams = line2ngrams(line)
        all_ngrams.append(ngams)

        for ngram in ngams:
            a, b = ngram
            if a in pairs_dct.keys():
                if b in pairs_dct[a]:
                    continue
                else:
                    pairs_dct[a].append(b)
            else:
                pairs_dct.update({a: [b]})

    all_ngrams = list(chain.from_iterable(all_ngrams))
    #all_ngrams_bad = [anti_ngram(n_gram, pairs_dct, uniq_words) for n_gram in all_ngrams]

    # bar = ProgressBar()
    # for ngram in bar(all_ngrams):
    #
    # print("ok")
    pairs_dict = pairs_dct.copy()
    p = Pool(4)
    all_ngrams_bad = p.map(anti_ngram, all_ngrams)

    all_ngrams = np.asarray(all_ngrams).astype("int16")
    all_ngrams_bad = np.asarray(all_ngrams_bad).astype("int16")

    np.save(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "all_ngrams_no_freq_words.npy"), all_ngrams)
    np.save(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "all_ngrams_bad_no_freq_words.npy"), all_ngrams_bad)
    print("ok")