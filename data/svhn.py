#!/usr/bin/python
# author eson
import copy

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN

from data.data_dir import DataDir


class NoiseSVHN(SVHN):
    def __init__(self, root=DataDir().get_dir("svhn"), split="train", transform=None, download=True, noise_rate=0.2):

        super().__init__(root=root, split=split, transform=transform, download=download)

        # if noise_rate != None:
        self.noise_rate = noise_rate
        self.add_label_noise()

    def __getitem__(self, item):
        img, noise_target, target = self.data[item], self.noise_targets[item], self.labels[item]

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

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
        self.noise_targets = copy.deepcopy(self.labels)
        for idx in self.poison_idx:
            rand_choice_list = list(range(self.labels.min(), self.labels.max() + 1))
            rand_choice_list.remove(self.labels[idx])
            self.noise_targets[idx] = np.random.choice(rand_choice_list)


if __name__ == '__main__':
    transform = get_transform(db_name="svhn")
    train_noise_svhn = NoiseSVHN(
        split="train",
        transform=transform["train_transform"],
        download=True
    )
    train_loader = DataLoader(dataset=train_noise_svhn, batch_size=128, shuffle=True, drop_last=True)
    for i, batch in enumerate(train_loader):
        (img, noise_target, target, if_noise) = batch.values()
        print(f"Batch {i} th, noise_rate {sum(if_noise) / len(target):.2f}")
