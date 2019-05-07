import os
import json
from itertools import chain

if __name__ == '__main__':
    path2data = "/Users/anastasia/flowers102/cvpr2016_flowers"
    p2textc10 = "text_c10"

    _path = os.path.join(path2data, p2textc10)
    all_folders = [os.path.join(_path, name) for name in os.listdir(_path) if os.path.isdir(os.path.join(_path, name))]

    all_text_files = list(chain.from_iterable(list(map(lambda x: [os.path.join(x, name) for name in os.listdir(x)
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

    freq_dict = {}
    for w in corpus_worlds:
        if w in freq_dict.keys():
            freq_dict[w] += 1
        else:
            freq_dict.update({w: 1})

    # with open("/Users/anastasia/PycharmProjects/diploma/outputs/words_frequency.json", "w") as fp:
    #     json.dump(freq_dict, fp)

    freq = list(sorted(list(freq_dict.items()), key=lambda x: x[1], reverse=True))
    with open("/Users/anastasia/PycharmProjects/diploma/outputs/freq_list.txt", "w") as fp:
        for item in freq:
            fp.write("{} {} \n".format(item[0], str(item[1])))
    fp.close()
    print("ok")
