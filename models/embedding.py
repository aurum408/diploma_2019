import os
import numpy as np
import torch
import torch.nn as nn

uniq_words = 6385


class W2Vec(torch.nn.Module):
    def __init__(self, voc_size, emb_size, p2emb=None):
        super(W2Vec, self).__init__()
        # self.voc_size = voc_size
        # self.emb_dim = emb_size
        if p2emb is None:
            self.embedding = nn.Embedding(voc_size, emb_size)
        else:
            self.embedding = torch.load(p2emb)

    def forward(self, x):
        x0_emb = self.embedding(x[0])
        x1_emb = self.embedding(x[1])
        cos_distance = torch.div(torch.sum(x0_emb * x1_emb, dim=-1),
                                 (torch.norm(x0_emb, dim=-1) * torch.norm(x1_emb, dim=-1)))

        return cos_distance

    def predict(self, x):
        return self.embedding(x)


def int2one_hot(int_inp, voc_size):
    def inp2onehot(inp, vs, sh):
        one_hot = np.zeros((vs, sh))
        for i in range(sh):
            one_hot[inp[i], i] = 1
        return one_hot

    num_v = int_inp.shape[0]
    x0 = int_inp[:, 0]
    x1 = int_inp[:, 1]
    x0_one_hot = [inp2onehot(x0[i], voc_size, num_v) for i in range(x0.shape[0])]
    x1_one_hot = [inp2onehot(x1[i], voc_size, num_v) for i in range(x1.shape[1])]

    return x0_one_hot, x1_one_hot


def int2onehot(ngams, u_words):
    num_ngrams = ngams.shape[0]
    ngrams_onehot = np.zeros((num_ngrams, 2, u_words))

    # bar = progressbar.ProgressBar()
    for i in range(num_ngrams):
        # print(i)
        idx0 = ngams[i][0]
        idx1 = ngams[i][1]
        ngrams_onehot[i][0][idx0] = 1
        ngrams_onehot[i][1][idx1] = 1
    return ngrams_onehot


def w2v_gen(true_examples, bad_examples, bs, voc_size=6385, onehot=False):
    num_ex = true_examples.shape[0]
    while True:
        item_s = bs // 2
        for j in range(0, num_ex, item_s):
            true_inp = true_examples[j:j + item_s]
            bad_inp = bad_examples[j:j + item_s]
            sh = bad_inp.shape[0]
            true_label = np.zeros((sh))
            bad_label = np.ones((sh))
            x = np.concatenate([true_inp, bad_inp])
            y = np.concatenate([true_label, bad_label])

            if onehot:
                x_onehot = int2onehot(x, voc_size)
                x0 = x_onehot[:, 0, :]
                x1 = x_onehot[:, 1, :]
            else:
                x_onehot = x
                x0 = x_onehot[:, 0]
                x1 = x_onehot[:, 1]

            yield [x0, x1], y


if __name__ == '__main__':
    all_2grams = np.load("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams.npy")
    all_2grams_bad = np.load("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams_bad.npy")

    gen = w2v_gen(all_2grams, all_2grams_bad, bs=1000, onehot=False)

    net = W2Vec(uniq_words, 100, "/Users/anastasia/PycharmProjects/diploma/outputs/model_200")
    # torch.save(net.embedding, "/Users/anastasia/PycharmProjects/diploma/outputs/model")
    criterion = torch.nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    lr = 1e-3
    optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
    # emb = nn.Embedding(6385, 100)
    # print("ok")
    # for i in range(10):
    #     x, y = next(gen)
    #     print(x)
    #     print("ok")

    from matplotlib import pyplot as plt

    all_loss = []
    all_loss_mean = []
    for j in range(300):
        for i in range(2047):
            x, y = next(gen)
            x0, x1 = x
            pred = net([torch.tensor(x0, dtype=torch.long), torch.tensor(x1, dtype=torch.long)])

            try:
                loss = criterion(pred, torch.tensor(y, dtype=torch.float))
            except:
                print("err")
                continue

            # print(loss.data.numpy())
            all_loss.append(loss.data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        all_loss_mean.append(np.mean(all_loss))
        all_loss = []

        plt.figure()
        plt.plot(all_loss_mean)
        # plt.show()
        plt.savefig("/Users/anastasia/PycharmProjects/diploma/outputs/history_804/loss_adagrad.png")
        plt.close()

        if j % 50 == 0:
            torch.save(net.embedding, "/Users/anastasia/PycharmProjects/diploma/outputs/model_804_{}".format(str(j)))
        if j % 30 == 0 and j > 0:
            lr = lr * 0.1
            print("optimizer se to {}".format(lr))
            optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
