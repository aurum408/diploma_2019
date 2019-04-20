import os
import numpy as np
from models.embedding import W2Vec, w2v_gen
import torch

if __name__ == '__main__':
    num_u = 6376

    all_2grams = np.load("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams_no_freq_words.npy")
    all_2grams_bad = np.load("/Users/anastasia/PycharmProjects/diploma/outputs/all_ngrams_bad_no_freq_words.npy")

    gen = w2v_gen(all_2grams, all_2grams_bad, bs=1000, onehot=False)

    net = W2Vec(num_u, 100, "/Users/anastasia/PycharmProjects/diploma/outputs/history_no_f_w_from250/model_904_150")

    criterion = torch.nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    lr = 1e-2
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

    p2save = "/Users/anastasia/PycharmProjects/diploma/outputs/history_no_f_w_from250_1"

    if not os.path.exists(p2save):
        os.makedirs(p2save)

    for j in range(200):
        for i in range(1074):
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
        plt.savefig(os.path.join(p2save, "loss_adagrad.png"))
        plt.close()

        if j % 50 == 0:
            torch.save(net.embedding, os.path.join(p2save, "model_904_{}".format(str(j))))
        if j == 199:
            torch.save(net.embedding, os.path.join(p2save, "model_904_{}".format(str(j))))
        # if j % 250 == 0 and j > 0:
        #     lr = lr * 0.1
        #     print("optimizer se to {}".format(lr))
        #     optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)