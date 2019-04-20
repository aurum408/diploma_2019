import os
import json
from itertools import chain


def text2ints(fpath, corpus_dict):
    _id = fpath.split("/")[-1].split(".txt")[0]
    with open(fpath, "r") as fp:
        lines = fp.readlines()
    lines = [item.split("\n")[0].split(' ') for item in lines]
    lines = list(map(lambda x: [_.split(".")[0] for _ in x], lines))
    lines = list(map(lambda x: [_.split(",")[0] for _ in x], lines))

    lines_int = list(map(lambda x:[corpus_dict[_] for _ in x], lines))
    return {_id: lines_int}

def ints2text(ints_line, c_dict):
    line = [c_dict[i] for i in ints_line]
    return line

if __name__ == '__main__':

    path2data = "/Users/anastasia/flowers102/cvpr2016_flowers"
    p2textc10 = "text_c10"

    _path = os.path.join(path2data, p2textc10)
    all_folders = [os.path.join(_path,name) for name in os.listdir(_path) if os.path.isdir(os.path.join(_path,name))]

    all_text_files = list(chain.from_iterable(list(map(lambda x:[os.path.join(x, name) for name in os.listdir(x)
                                                                 if name.endswith(".txt")], all_folders))))

    corpus = []
    for file in all_text_files:
        with open(file, "r") as fp:
            corpus.append(fp.readlines())
    corpus = list(chain.from_iterable(corpus))
    corpus = [item.split("\n")[0].split(' ') for item in corpus]

    corpus_worlds = list(chain.from_iterable(corpus))
    corpus_worlds = [w.split(".")[0] for w in corpus_worlds]
    corpus_worlds = [w.split(",")[0] for w in corpus_worlds]

    corpus_dict = {}
    uniq_words = 0

    for world in corpus_worlds:
        if world not in corpus_dict.keys():
            corpus_dict.update({world: uniq_words})
            uniq_words += 1

    encoded = text2ints(all_text_files[0], corpus_dict)
    reversed = {v:k for k, v in corpus_dict.items()}

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "corpus_word2int.json"), "w") as fp:
        json.dump(corpus_dict, fp)

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "corpus_int2word.json"), "w") as fp:
        json.dump(reversed, fp)

    print('num_uniq_w')
