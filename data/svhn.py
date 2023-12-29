#!/usr/bin/python
# author eson
import collections
import copy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from tqdm import tqdm

from data.data_dir import DataDir

device = "cuda" if torch.cuda.is_available() else "cpu"


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


class PoisonNoiseSVHN(NoiseSVHN):
    def __init__(self, root, split='train', transform=None, noise_rate=0.2, noise_idx=None,
                 download=False, poison_rate=1.0, perturbfile_path=None, hl_perturbfile_path=None,
                 perturb_type='samplewise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonNoiseSVHN, self).__init__(root=root, split=split, download=download, transform=transform,
                                              noise_rate=noise_rate)
        self.perturb_tensor = torch.load(perturbfile_path, map_location=device)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).to('cpu').numpy()
        if hl_perturbfile_path is not None:
            self.hl_perturb_tensor = torch.load(hl_perturbfile_path, map_location=device)
            self.hl_perturb_tensor = self.hl_perturb_tensor.mul(255).clamp_(0, 255).to('cpu').numpy()

        self.patch_location = patch_location
        self.img_denoise = img_denoise
        # Check Shape
        # if perturb_type == 'samplewise' and (self.perturb_tensor.shape[0]+self.hl_perturb_tensor.shape[0]) != len(self):
        #     raise ('Poison Perturb Tensor size not match for samplewise')
        # elif perturb_type == 'classwise' and self.perturb_tensor.shape[0] != 10:
        #     raise ('Poison Perturb Tensor size not match for classwise')

        self.data = self.data.astype(np.float32)

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if noise_idx is not None:
            self.poison_samples_idx = torch.arange(len(self))[
                torch.where(torch.isin(torch.arange(len(self)), noise_idx) == False, True, False)]
        else:
            self.poison_samples_idx=torch.arange(len(self))
        # low loss samples perturbation
        for i,idx in tqdm(enumerate(self.poison_samples_idx)):
            self.poison_samples[idx] = True
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = self.perturb_tensor[i]
                # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = self.perturb_tensor[self.labels[idx]]
                # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

            if add_uniform_noise:
                noise = np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] += noise
            self.data[idx] = np.clip(self.data[idx], 0, 255)

        # high loss samples perturbation
        if hl_perturbfile_path is not None:

            for i,idx in tqdm(enumerate(noise_idx)):
                self.poison_samples[idx] = True
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = self.hl_perturb_tensor[i]
                    # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = self.hl_perturb_tensor[self.labels[idx]]
                    # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

                if add_uniform_noise:
                    noise = np.random.uniform(0, 8, (32, 32, 3))

                self.data[idx] += noise
                self.data[idx] = np.clip(self.data[idx], 0, 255)

        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


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
