#!/usr/bin/python
# author eson
import collections
import copy
import os
import pickle
from typing import Any, Optional, Callable, Tuple
import torch
import numpy as np
from PIL import Image
import random
from torchvision.datasets import SVHN, VisionDataset, CIFAR100
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from tqdm import tqdm

from data.data_dir import DataDir
from utils.common_utils import patch_noise_extend_to_img

device = "cuda" if torch.cuda.is_available() else "cpu"


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class PoisonCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None,
                 download=False, poison_rate=1.0, perturbfile_path=None,
                 perturb_type='samplewise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonCIFAR10, self).__init__(root=root, train=train, download=download,
                                            transform=transform, )
        self.perturb_tensor = torch.load(perturbfile_path, map_location=device)

        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[
            1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise ('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise ('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        targets = list(range(0, len(self)))
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(
                    np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(
                np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        for i, idx in tqdm(enumerate(self.poison_samples_idx)):
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[i]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class NoiseCIFAR10(CIFAR10):
    def __init__(self, root=DataDir().get_dir("cifar-10"), train=True, transform=None, download=True, noise_rate=0.2):

        super().__init__(root=root, train=train, transform=transform, download=download)

        # if noise_rate != None:
        self.noise_rate = noise_rate
        self.add_label_noise()
        self.return_index = False

    def __getitem__(self, item):
        img, noise_target, target = self.data[item], self.noise_targets[item], self.targets[item]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.return_index:
            return {"image": img,
                    "noise_target": noise_target,
                    "true_target": target,
                    "if_noise": self.if_noise[item],
                    "item": item}
        return {"image": img,
                "noise_target": noise_target,
                "true_target": target,
                "if_noise": self.if_noise[item]}

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


class PoisonNoiseCIFAR10(NoiseCIFAR10):
    def __init__(self, root, train=True, transform=None, noise_rate=0.2, noise_idx=None,
                 download=False, poison_rate=1.0, perturbfile_path=None, hl_perturbfile_path=None,
                 perturb_type='samplewise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonNoiseCIFAR10, self).__init__(root=root, train=train, download=download,
                                                 transform=transform, noise_rate=noise_rate)
        if perturbfile_path:
            self.perturb_tensor = torch.load(perturbfile_path, map_location=device)
            if len(self.perturb_tensor.shape) == 4:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            else:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to(
                    'cpu').numpy()
        self.patch_location = patch_location
        self.data = self.data.astype(np.float32)

        # if perturb_type == 'samplewise' and target_dim != len(self):
        #     raise ('Poison Perturb Tensor size not match for samplewise')
        # elif perturb_type == 'classwise' and target_dim != 10:
        #     raise ('Poison Perturb Tensor size not match for classwise')
        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []

        # get_poison_sample_idx
        # self.get_poison_samples_idx(noise_idx,poison_rate=1.0)
        if noise_idx is not None:

            self.poison_samples_idx = torch.arange(50000)[
                torch.where(torch.isin(torch.arange(50000), noise_idx) == False, True, False)]
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(
                np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        if perturbfile_path:
            for i, idx in tqdm(enumerate(self.poison_samples_idx)):
                self.poison_samples[idx] = True
                if len(self.perturb_tensor.shape) == 5:
                    perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                    perturb_tensor = self.perturb_tensor[perturb_id]
                else:
                    perturb_tensor = self.perturb_tensor
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = perturb_tensor[i]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = perturb_tensor[self.noise_targets[idx]]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                self.data[idx] = self.data[idx] + noise
                self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)

        if hl_perturbfile_path:
            self.hl_perturb_tensor = torch.load(hl_perturbfile_path, map_location=device)
            if len(self.hl_perturb_tensor.shape) == 4:
                self.hl_perturb_tensor = self.hl_perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to(
                    'cpu').numpy()
            else:
                self.hl_perturb_tensor = self.hl_perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to(
                    'cpu').numpy()
            # high loss samples perturbation
            for i, idx in tqdm(enumerate(noise_idx)):
                self.poison_samples[idx] = True
                if len(self.hl_perturb_tensor.shape) == 5:
                    hl_perturb_id = random.choice(range(self.hl_perturb_tensor.shape[0]))
                    hl_perturb_tensor = self.hl_perturb_tensor[hl_perturb_id]
                else:
                    hl_perturb_tensor = self.hl_perturb_tensor
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = hl_perturb_tensor[i]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

                self.data[idx] = self.data[idx] + noise
                self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)

        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def poison_classwise(self, noise_idx):

        for i, label in enumerate(noise_idx):
            self.poison_samples_idx.append(i)


class NoiseCIFAR100(CIFAR100):
    def __init__(self, root=DataDir().get_dir("cifar-100"), train=True, transform=None, download=True, noise_rate=0.2):

        super().__init__(root=root, train=train, transform=transform, download=download)

        # if noise_rate != None:
        self.noise_rate = noise_rate
        self.add_label_noise()
        self.return_index = False

    def __getitem__(self, item):
        img, noise_target, target = self.data[item], self.noise_targets[item], self.targets[item]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.return_index:
            return {"image": img,
                    "noise_target": noise_target,
                    "true_target": target,
                    "if_noise": self.if_noise[item],
                    "item": item}
        return {"image": img,
                "noise_target": noise_target,
                "true_target": target,
                "if_noise": self.if_noise[item]}

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


class PoisonNoiseCIFAR100(NoiseCIFAR100):
    def __init__(self, root, train=True, transform=None, noise_rate=0.2, noise_idx=None,
                 download=False, poison_rate=1.0, perturbfile_path=None, hl_perturbfile_path=None,
                 perturb_type='samplewise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonNoiseCIFAR100, self).__init__(root=root, train=train, download=download,
                                                  transform=transform, noise_rate=noise_rate)
        if perturbfile_path:
            self.perturb_tensor = torch.load(perturbfile_path, map_location=device)
            if len(self.perturb_tensor.shape) == 4:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            else:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to(
                    'cpu').numpy()
        self.patch_location = patch_location
        self.data = self.data.astype(np.float32)

        # if perturb_type == 'samplewise' and target_dim != len(self):
        #     raise ('Poison Perturb Tensor size not match for samplewise')
        # elif perturb_type == 'classwise' and target_dim != 10:
        #     raise ('Poison Perturb Tensor size not match for classwise')
        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []

        # get_poison_sample_idx
        # self.get_poison_samples_idx(noise_idx,poison_rate=1.0)
        if noise_idx is not None:
            self.poison_samples_idx = torch.arange(50000)[
                torch.where(torch.isin(torch.arange(50000), noise_idx) == False, True, False)]
        else:
            self.poison_samples_idx = torch.arange(50000)
        if perturbfile_path:
            for i, idx in tqdm(enumerate(self.poison_samples_idx)):
                self.poison_samples[idx] = True
                if len(self.perturb_tensor.shape) == 5:
                    perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                    perturb_tensor = self.perturb_tensor[perturb_id]
                else:
                    perturb_tensor = self.perturb_tensor
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = perturb_tensor[i]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = perturb_tensor[self.noise_targets[idx]]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                self.data[idx] = self.data[idx] + noise
                self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)

        if hl_perturbfile_path:
            self.hl_perturb_tensor = torch.load(hl_perturbfile_path, map_location=device)
            if len(self.hl_perturb_tensor.shape) == 4:
                self.hl_perturb_tensor = self.hl_perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to(
                    'cpu').numpy()
            else:
                self.hl_perturb_tensor = self.hl_perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to(
                    'cpu').numpy()
            # high loss samples perturbation
            for i, idx in tqdm(enumerate(noise_idx)):
                self.poison_samples[idx] = True
                if len(self.hl_perturb_tensor.shape) == 5:
                    hl_perturb_id = random.choice(range(self.hl_perturb_tensor.shape[0]))
                    hl_perturb_tensor = self.hl_perturb_tensor[hl_perturb_id]
                else:
                    hl_perturb_tensor = self.hl_perturb_tensor
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = hl_perturb_tensor[i]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

                self.data[idx] = self.data[idx] + noise
                self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)

        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def poison_classwise(self, noise_idx):

        for i, label in enumerate(noise_idx):
            self.poison_samples_idx.append(i)
