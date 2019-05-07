import pickle
import torch
import h5py as h5
from progressbar import progressbar


def load_py2(pth):
    pickle_in = open(pth, "rb")
    out = pickle.load(pickle_in, encoding='bytes')
    return out


if __name__ == '__main__':
    pth1 = "/hdd1/flowers102/char-cnn-rnn-preprocessed/train/char-CNN-RNN-embeddings.pickle"
    pth2 = "/hdd1/flowers102/char-cnn-rnn-preprocessed/train/class_info.pickle"
    pth3 = "/hdd1/flowers102/char-cnn-rnn-preprocessed/train/filenames.pickle"

    preprocessed_train = load_py2(pth1)
    class_info_train = load_py2(pth2)
    filenames_train = load_py2(pth3)

    pth1 = "/hdd1/flowers102/char-cnn-rnn-preprocessed/test/char-CNN-RNN-embeddings.pickle"
    pth2 = "/hdd1/flowers102/char-cnn-rnn-preprocessed/test/class_info.pickle"
    pth3 = "/hdd1/flowers102/char-cnn-rnn-preprocessed/test/filenames.pickle"

    preprocessed_test = load_py2(pth1)
    class_info_test = load_py2(pth2)
    filenames_test = load_py2(pth3)

    all_preprocessed = preprocessed_train + preprocessed_test
    all_class_info = class_info_train + class_info_test
    all_filenames = filenames_train + filenames_test

    p2save = "/hdd1/diploma_outputs/outputs/char-cnn-rnn-emb.hdf5"

    f = h5.File(p2save, "w")
    embeddings = f.create_group("embeddings")

    bar = progressbar.ProgressBar()
    names = list(zip(all_filenames, all_preprocessed))
    for name, data in bar(names):
        name = name.decode("utf-8")
        dset_name = name.split("/")[-1]
        embeddings.create_dataset(name=dset_name, data=data)
    f.close()
    print("ok")
