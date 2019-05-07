import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

if __name__ == '__main__':
    p2images = "/hdd1/flowers102/"

    imageSize = 256
    batchSize = 64
    workers = 4
    dataset = dset.ImageFolder(root=p2images,
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor()
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=int(workers))

    print("ok")

    for i, data in enumerate(dataloader, 0):
        if i < 10:
            sample = data[0].numpy()
            sample = sample * 255.
            print("ok")

        else:
            break
