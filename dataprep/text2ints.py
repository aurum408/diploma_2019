import os
import json

if __name__ == '__main__':
    with open("/Users/anastasia/PycharmProjects/diploma/outputs/corpus_word2int_no_freq_w.json", "r") as fp:
        corpus = json.load(fp)

    with open("/Users/anastasia/PycharmProjects/diploma/outputs/corpus_int2word_no_freq_w.json", "r") as fp:
        corpus_reversed = json.load(fp)

    path2data = "/Users/anastasia/flowers102/cvpr2016_flowers"
    p2textc10 = "text_c10"

    _path = os.path.join(path2data, p2textc10)
    all_folders = [os.path.join(_path, name) for name in os.listdir(_path) if os.path.isdir(os.path.join(_path, name))]

    all_text = {}

    freq_words = ["flower", "petals", "and", "this", "that", "and", "has", "are", 'a', "the"]

    for folder in all_folders:
        class_id = folder.split('/')[-1]
        txt_files = [item for item in os.listdir(folder) if item.endswith(".txt")]

        class_dct = {}

        for item in txt_files:
            image_id = item.split(".txt")[0]

            with open(os.path.join(folder, item), "r") as fp:
                lines = fp.readlines()

            lines = [item.split("\n")[0].split(' ') for item in lines]

            lines = list(map(lambda x: [w.split(".")[0] for w in x], lines))
            lines = list(map(lambda x: [w.split(",")[0] for w in x], lines))

            lines2int = list(map(lambda x: [corpus[w] for w in x if w not in freq_words], lines))

            class_dct.update({image_id: lines2int})
        all_text.update({class_id: class_dct})

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "all_text2ints_no_freq_w.json"),
              "w") as fp:
        json.dump(all_text, fp)
    print("ok")
