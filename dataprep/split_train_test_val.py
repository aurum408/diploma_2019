import os
import numpy as np
import json

seed = 8735163
train_test_val_split = "90/10/10"

train_p = 0.8
test_p = 0.1
val_p = 0.1

if __name__ == '__main__':
    path2data = "/Users/anastasia/flowers102/cvpr2016_flowers"
    p2textc10 = "text_c10"

    _path = os.path.join(path2data, p2textc10)
    all_folders = [os.path.join(_path, name) for name in os.listdir(_path) if os.path.isdir(os.path.join(_path, name))]

    train_data = {}
    test_data = {}
    val_data = {}

    class_info = {}

    for folder in all_folders:
        class_id = folder.split("/")[-1]
        f_ids = [item.split('.txt')[0] for item in os.listdir(folder) if item.endswith(".txt")]
        np.random.seed(seed)
        np.random.shuffle(f_ids)
        num_files = len(f_ids)
        train_ids = f_ids[:int(num_files * train_p)]
        test_ids = f_ids[int(num_files * train_p):int(num_files * train_p) + int(num_files * test_p)]
        val_ids = f_ids[num_files - int(num_files * val_p):]

        train_data.update({class_id: train_ids})
        test_data.update({class_id: test_ids})
        val_data.update({class_id: val_ids})

        class_info.update({class_id: num_files})

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "train_data.json"), "w") as fp:
        json.dump(train_data, fp)

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "test_data.json"), "w") as fp:
        json.dump(test_data, fp)

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "val_data.json"), "w") as fp:
        json.dump(val_data, fp)

    with open(os.path.join("/Users/anastasia/PycharmProjects/diploma/outputs", "num_items_per_class.json"), "w") as fp:
        json.dump(class_info, fp)

    print("ok")
