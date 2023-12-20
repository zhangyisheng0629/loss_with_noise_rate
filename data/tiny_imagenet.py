#!/usr/bin/python
# author eson
import copy

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms

from data.data_dir import DataDir


class TinyImageNet(ImageNet):
    def __init__(self, root=DataDir().get_dir("imagenet"), split="train", transform=None):
        super().__init__(root=root, split=split, transform=transform)


class NoiseTinyImageNet(TinyImageNet):
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


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.CenterCrop(256),
         transforms.Resize((32, 32)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    train_noise_imagenet = NoiseTinyImageNet(split="train", transform=transform)
    train_loader = DataLoader(dataset=train_noise_imagenet, batch_size=128, shuffle=True, drop_last=True,
                              num_workers=20)
    for i, batch in enumerate(train_loader):
        (img, noise_target, target, if_noise) = batch.values()
        print(f"Batch {i} th, noise_rate {sum(if_noise) / len(target):.2f}")
