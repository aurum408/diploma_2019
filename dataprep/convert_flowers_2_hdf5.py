import os
import numpy as np
import json
import h5py as h5
from PIL import Image
from progressbar.progressbar import ProgressBar


def data2hdf5(class_dct, emb_data, p2img, p2save, group_name):
    bar = ProgressBar()
    f = h5.File(p2save, "w")
    group = f.create_group(name=group_name)
    for class_name, items_list in bar(class_dct.items()):
        class_group = group.create_group(name=class_name)
        for item in items_list:
            img = Image.open(os.path.join(p2img, item + ".jpg"))
            img = np.asarray(img, dtype="uint8")
            emb = emb_data["embeddings"][item][:]

            item_group = class_group.create_group(name=item)
            item_group.create_dataset(name="image", data=img)
            item_group.create_dataset(name="embeddings", data=emb)
    f.close()


if __name__ == '__main__':
    p2emb = "/hdd1/diploma_outputs/outputs/char-cnn-rnn-emb.hdf5"
    p2data = "/hdd1/diploma_outputs/outputs"
    p2images = "/hdd1/flowers102/jpg"

    embeddings = h5.File(p2emb, "r")
    p2save = "/hdd1/diploma_outputs/outputs"

    with open(os.path.join(p2data, "train_data_dj.json"), "r") as fp:
        train_classes = json.load(fp)

    data2hdf5(train_classes, embeddings, p2images, os.path.join(p2save, "train_flowers.hdf5"), group_name="train")

    with open(os.path.join(p2data, "test_data_dj.json"), "r") as fp:
        test_classes = json.load(fp)

    data2hdf5(test_classes, embeddings, p2images, os.path.join(p2save, "test_flowers.hdf5"), group_name="test")

    with open(os.path.join(p2data, "val_data_dj.json"), "r") as fp:
        val_classes = json.load(fp)

    data2hdf5(val_classes, embeddings, p2images, os.path.join(p2save, "val_flowers.hdf5"), group_name="val")

    embeddings.close()
