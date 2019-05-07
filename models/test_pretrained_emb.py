import torchfile
import torch
# from torch.utils.serialization import load_lua
import json

if __name__ == '__main__':
    pth = "/Users/anastasia/PycharmProjects/diploma/outputs/num_items_per_class.json"

    with open(pth, "r") as fp:
        data = json.load(fp)

    data = list(sorted([(k, v) for k, v in data.items()], key=lambda x: x[1], reverse=True))
    print("ok")
