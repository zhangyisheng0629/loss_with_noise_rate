#!/usr/bin/python
# author eson
import copy

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from data.data_dir import DataDir
import torchvision.transforms as transforms


class NoiseCIFAR10(CIFAR10):
    def __init__(self, root=DataDir().get_dir("cifar-10"), train=True, transform=None, download=True, noise_rate=0.2):

        CIFAR10.__init__(self, root=root, train=train, transform=transform, download=download)

        # if noise_rate != None:
        self.noise_rate = noise_rate
        self.add_label_noise()

    def __getitem__(self, item):
        img, noise_target, target = self.data[item], self.noise_targets[item], self.targets[item]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return {"image": img,
                "noise_target": noise_target,
                "true_target": target,
                "if_noise ": self.if_noise[item]}

    def __len__(self):
        return len(self.data)

    def add_label_noise(self):
        self.poison_idx = (np.random.choice(len(self), int(len(self) * self.noise_rate), replace=False))
        self.poison_idx.sort()
        self.if_noise = [True if i in self.poison_idx else False for i in range(len(self))]
        self.noise_targets = copy.deepcopy(self.targets)
        for idx in self.poison_idx:
            rand_choice_list = list(self.class_to_idx.values())
            rand_choice_list.remove(self.targets[idx])
            self.noise_targets[idx] = np.random.choice(rand_choice_list)


if __name__ == '__main__':
    from utils.trans import get_transform

    transform = get_transform("cifar-10")
    noise_cifar10 = NoiseCIFAR10(train=True, transform=transform["train_transform"], download=True)
    train_loader = DataLoader(dataset=noise_cifar10, batch_size=128, shuffle=True, drop_last=True)
    for i, batch in enumerate(train_loader):
        (img, noise_target, target, if_noise) = batch.values()
        print(f"Batch {i} th, noise_rate {sum(if_noise) / len(target):.2f}")
