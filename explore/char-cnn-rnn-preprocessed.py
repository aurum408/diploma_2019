import pickle
import torch


def load_py2(pth):
    pickle_in = open(pth, "rb")
    out = pickle.load(pickle_in, encoding='bytes')
    return out

if __name__ == '__main__':
    pth1 = "/Users/anastasia/flowers102/char-cnn-rnn-preprocessed/train/char-CNN-RNN-embeddings.pickle"
    pth2 = "/Users/anastasia/flowers102/char-cnn-rnn-preprocessed/train/class_info.pickle"
    pth3 = "/Users/anastasia/flowers102/char-cnn-rnn-preprocessed/train/filenames.pickle"

    preprocessed = load_py2(pth1)
    class_info = load_py2(pth2)
    filenames = load_py2(pth3)


    print("ok")