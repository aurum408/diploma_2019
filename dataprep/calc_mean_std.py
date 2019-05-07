import os
import h5py as h5
import numpy as np
from functools import reduce


def calc_mean(dset, num_pixels):
    all_sum = [np.sum(item[:], axis=(0, 1)) for item in dset]
    all_sum = np.stack(all_sum, axis=0)
    all_sum = np.sum(all_sum, axis=0)
    mean = all_sum / num_pixels
    return mean


def calc_std(dset, num_pixel, mean):
    all_sum = [np.sum(np.square(item[:] - mean), axis=(0, 1)) for item in dset]
    all_sum = np.stack(all_sum, axis=0)
    all_sum = np.sum(all_sum, axis=0)
    std = np.sqrt(all_sum / num_pixels)
    return std


if __name__ == '__main__':
    p2data = "/hdd1/diploma_outputs/outputs"
    dset_name = "train_flowers.hdf5"

    f = h5.File(os.path.join(p2data, dset_name))
    class_names = f["train"].keys()
    data = f["train"]

    image_arr = []
    for name in class_names:
        image_names = data[name]
        for img_name in image_names:
            image_arr.append(data[name][img_name]["image"])

    shapes = [img.shape[:-1] for img in image_arr]
    num_pixels = list(map(lambda x: x[0] * x[1], shapes))
    num_pixels = reduce(lambda x, y: x + y, num_pixels)
    print("ok")

    # mean = calc_mean(image_arr, num_pixels)
    mean = [109.54635135, 95.23376605, 70.32107908]
    print(mean)
    std = calc_std(image_arr, num_pixels, mean)
    print(std)
