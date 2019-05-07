import os
from utils.txt2image_dataset import Text2ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.dcgan import Generator, Discriminator, weights_init
from train_gan import Trainer


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

    dset = Text2ImageDataset(os.path.join(p2data, dset_name), split="train", transform=transform, mean=0, std=1)

    gen = Generator()
    discr = Discriminator()




    # gen, discr, type, dataset, lr, diter, vis_screen, save_path, l1_coef, l2_coef,
    # pre_trained_gen,
    # pre_trained_disc, batch_size, num_workers, epochs
    #output2 - custom initialization
    #output3 - default initialization

    gan_trainer = Trainer(gen, discr, dset, 0.0002, "gan4", "output3", 50, 100, False, False, 64, 4, 101)
    gan_trainer.train(cls=True, spe=125)