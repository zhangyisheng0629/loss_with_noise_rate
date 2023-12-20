#!/usr/bin/python
# author eson
import copy
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
import time
import multiprocessing

from data.data_dir import DataDir


# from utils.get import get_transform


class NoiseImageNet(ImageNet):
    def __init__(self, root=DataDir().get_dir("imagenet"), split="train", transform=None, noise_rate=0.2):
        super().__init__(root=root, split=split, transform=transform)
        self.noise_rate = noise_rate
        self.rand_choice_list = list(self.wnid_to_idx.values())
        self.poison_idx = (np.random.choice(len(self), int(len(self) * self.noise_rate), replace=False))
        self.poison_idx.sort()
        self.if_noise = [True if i in self.poison_idx else False for i in range(len(self))]

    def __getitem__(self, item):
        # img:str, target:int
        path, target = self.imgs[item]
        img = self.loader(path)
        # noise_target = self.noise_targets[item],
        if self.if_noise[item]:
            noise_target = self.single_noise(item)
        else:
            noise_target = target

        if self.transform is not None:
            img = self.transform(img)
        return {"image": img,
                "noise_target": noise_target,
                "true_target": target,
                "if_noise ": self.if_noise[item]}

    def __len__(self):
        return len(self.imgs)

    def single_noise(self, idx):
        rand_choice_list_ = copy.deepcopy(self.rand_choice_list)
        rand_choice_list_.remove(self.targets[idx])

        noise_target = np.random.choice(rand_choice_list_)
        return noise_target

    def add_label_noise(self):
        self.poison_idx = (np.random.choice(len(self), int(len(self) * self.noise_rate), replace=False))
        self.poison_idx.sort()
        self.if_noise = [True if i in self.poison_idx else False for i in range(len(self))]
        self.noise_targets = copy.deepcopy(self.targets)
        # serial
        # for idx in self.poison_idx:

        # parallel
        # s = time.time()
        # pool = multiprocessing.Pool()
        # pool.map(self.single_noise, self.poison_idx)
        # # print(f"d2:{time.time() - s}", end="")
        # pool.close()
        # pool.join()
        # d2 = time.time() - s

        # print(d1,d2)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor()
    ])
    train_noise_imagenet = NoiseImageNet(split="train", transform=transform)
    train_loader = DataLoader(dataset=train_noise_imagenet, batch_size=128, shuffle=True, drop_last=True,
                              num_workers=10)
    for i, batch in enumerate(train_loader):
        (img, noise_target, target, if_noise) = batch.values()
        print(f"Batch {i} th, noise_rate {sum(if_noise) / len(target):.2f}")
