import os
import numpy as np
import json

seed = 8735163
train_test_val_split = "90/10/10"

train_p=0.8
test_p=0.1
val_p=0.1


def collect_files(class_list, pth):
    dct = {}
    for item in class_list:
        fnames = os.listdir(os.path.join(pth, item))
        fnames = [item.split('.txt')[0] for item in fnames if item.endswith(".txt")]
        dct.update({item: fnames})
    return dct


if __name__ == '__main__':
    path2data = "/Users/anastasia/flowers102/cvpr2016_flowers"
    p2textc10 = "text_c10"

    _path = os.path.join(path2data, p2textc10)
    all_folders = [os.path.join(_path, name) for name in os.listdir(_path) if os.path.isdir(os.path.join(_path, name))]

    train_data = {}
    test_data = {}
    val_data = {}

    class_info = {}

    class_ids = [folder.split("/")[-1] for folder in all_folders]

    np.random.seed(seed)
    np.random.shuffle(class_ids)

    num_classes = len(class_ids)

    train_cl = class_ids[:int(num_classes*train_p)]
    test_cl = class_ids[int(num_classes*train_p):(int(num_classes*train_p) + (int(num_classes*test_p)))]
    val_cl = class_ids[(int(num_classes*train_p)+ (int(num_classes*test_p))):]

    train_data = collect_files(train_cl, os.path.join(path2data, p2textc10))
    test_data = collect_files(test_cl, os.path.join(path2data, p2textc10))
    val_data = collect_files(val_cl, os.path.join(path2data, p2textc10))

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "train_data_dj.json"), "w") as fp:
        json.dump(train_data, fp)

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "test_data_dj.json"), "w") as fp:
        json.dump(test_data, fp)

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "val_data_dj.json"), "w") as fp:
        json.dump(val_data, fp)