import os
import io
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F


class Text2ImageDataset(Dataset):

    def __init__(self, datasetFile, transform=None, split="train", mean=None, std=None):
        self.datasetFile = datasetFile
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = split
        self.mean = mean
        self.std = std
        self.h5py2int = lambda x: int(np.array(x))

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()

        return length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]

        # pdb.set_trace()

        key = np.random.choice(list(example.keys()))
        right_image = example[key]['image'][:]
        right_embed = np.array(example[key]['embeddings'][np.random.randint(0,10)], dtype=float)
        wrong_image = self.find_wrong_image(example_name)
        # inter_embed = np.array(self.find_inter_embed())

        right_image = Image.fromarray(right_image)
        wrong_image = Image.fromarray(wrong_image)

        # right_image = self.validate_image(right_image)
        # wrong_image = self.validate_image(wrong_image)

        # txt = np.array(example['txt']).astype(str)

        sample = {
            'right_images': self.transform(right_image),
            'right_embed': torch.FloatTensor(right_embed),
            'wrong_images': self.transform(wrong_image),
            # 'inter_embed': torch.FloatTensor(inter_embed),
            # 'txt': str(txt)
        }

        # sample['right_images'] = sample['right_images'].sub_(self.mean).div_(self.std)
        # sample['wrong_images'] =sample['wrong_images'].sub_(self.mean).div_(self.std)

        return sample

    def find_wrong_category(self, category):
        idx = self.dataset_keys.index(category)
        names = self.dataset_keys.copy()
        del names[idx]
        example_name = np.random.choice(names)
        return example_name

    def find_wrong_image(self, category):
        example_name = self.find_wrong_category(category)
        key = np.random.choice(list(self.dataset[self.split][example_name].keys()))
        example = self.dataset[self.split][example_name][key]["image"][:]
        return example

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)
