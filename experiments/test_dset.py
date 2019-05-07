import os
from utils.txt2image_dataset import Text2ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from time import time

ch_0_mean = 109.54635135
ch_1_mean = 95.23376605
ch_2_mean = 70.32107908

mean = [109.54635135, 95.23376605, 70.32107908]
std = [76.04231164, 62.36785183, 68.2234495]

if __name__ == '__main__':
    p2data = "/hdd1/diploma_outputs/outputs"
    dset_name = "train_flowers.hdf5"

    imageSize = 64

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=imageSize, scale=(0.9, 0.95), ratio=(1, 1)),
        transforms.ToTensor()])

    batchSize = 256
    workers = 4

    dset = Text2ImageDataset(os.path.join(p2data, dset_name), split="train", transform=transform, mean=mean, std=std)
    dataloader = DataLoader(dset, batch_size=batchSize,
                            shuffle=True, num_workers=int(workers))
    print("ok")

    i = 0
    for data in dataloader:
        print(i)
        i+=1
        if i < 10:
            t0 = time()
            sample = data
            print(time() - t0)
        #i+=1

    # else:
    # break
