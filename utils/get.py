#!/usr/bin/python
# author eson
# !/usr/bin/python
# author eson
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, SVHN, CIFAR100

from data.cifar import CIFAR10, PoisonCIFAR10, PoisonNoiseCIFAR10, NoiseCIFAR10, NoiseCIFAR100, PoisonNoiseCIFAR100
from data.data_dir import DataDir
from data.imagenet import NoiseImageNet

from data.svhn import NoiseSVHN, PoisonNoiseSVHN
from data.tiny_imagenet import NoiseTinyImageNet, TinyImageNet
from data.imagenet_mini import ImageNetMini, NoiseImageNetMini

def get_dataset_(stage, db_name, noise_rate=0.2, train_aug=True, noise_path=None, perturbfile_path=None,
                hl_perturbfile_path=None,perturb_type="samplewise",poison_rate=1.0):
    # transform
    transform = get_transform(db_name)
    if not train_aug:
        transform["train_transform"] = transform["val_transform"]

    noise_idx = torch.load(noise_path) if noise_path else None
    # dataset
    if stage == "select":
        train_dataset = get_train_dataset_stage12_(transform, db_name, noise_rate, )
    elif stage == "ue_gen":
        print("------ue_gen------")
        train_dataset = get_train_dataset_stage12_(transform, db_name, noise_rate, noise_idx=noise_idx)
    elif stage == "ue_train":
        print("------ue_train------")
        train_dataset = get_train_dataset_stage3_(transform, db_name, noise_rate, noise_idx=noise_idx,
                                                 perturbfile_path=perturbfile_path,
                                                 hl_perturbfile_path=hl_perturbfile_path,
                                                 perturb_type=perturb_type,
                                                  poison_rate=poison_rate)

    else:
        raise ValueError(f"Invalid stage name {stage}, set it from 'select, ue_gen, ue_train'. ")

    val_dataset = get_val_dataset(db_name, transform)

    return {"train": train_dataset, "val": val_dataset}


def get_train_dataset_stage12_(transform, db_name, noise_rate: float = 0, noise_idx=None):
    if db_name == "cifar-10":

        if noise_rate:
            train_dataset = NoiseCIFAR10(
                root=DataDir().get_dir("cifar-10"),
                train=True,
                transform=transform["train_transform"],
                download=True,
                noise_rate=noise_rate
            )
        else:
            train_dataset = CIFAR10(
                root=DataDir().get_dir("cifar-10"),
                train=True,
                transform=transform["train_transform"],
                download=True,
            )
    elif db_name == "cifar-100":

        if noise_rate:
            train_dataset = NoiseCIFAR100(
                root=DataDir().get_dir("cifar-100"),
                train=True,
                transform=transform["train_transform"],
                download=True,
                noise_rate=noise_rate
            )
        else:
            train_dataset = CIFAR100(
                root=DataDir().get_dir("cifar-100"),
                train=True,
                transform=transform["train_transform"],
                download=True,
            )

    elif db_name == "svhn":

        if noise_rate:
            train_dataset = NoiseSVHN(
                root=DataDir().get_dir("svhn"),
                split="train",
                transform=transform["train_transform"],
                download=True,
                noise_rate=noise_rate
            )
        else:
            train_dataset = SVHN(
                root=DataDir().get_dir("svhn"),
                split="train",
                transform=transform["train_transform"],
                download=True
            )
    elif db_name == "imagenet_mini":

        if noise_rate:
            train_dataset = NoiseImageNetMini(
                root=DataDir().get_dir("imagenet_mini"),
                split="train",
                transform=transform["train_transform"],

                noise_rate=noise_rate
            )
        else:
            train_dataset = ImageNetMini(
                root=DataDir().get_dir("imagenet_mini"),
                split="train",
                transform=transform["train_transform"],

            )
    else:

        raise NotImplementedError(f"No {db_name} train dataset exist.")
    if noise_idx != None:
        total_len = len(train_dataset)
        subset_idx = torch.arange(total_len)[
            torch.where(torch.isin(torch.arange(total_len), noise_idx) == False, True, False)]
        ll_dataset = Subset(train_dataset, subset_idx)
        hl_dataset = Subset(train_dataset, noise_idx)
        train_dataset = {"ll": ll_dataset, "hl": hl_dataset}
    else:
        train_dataset = train_dataset

    return train_dataset


def get_train_dataset_stage3_(transform, db_name, noise_rate: float = 0, noise_idx=None, perturbfile_path=None,
                             hl_perturbfile_path=None,perturb_type="samplewise",poison_rate=1.0):
    if db_name == "cifar-10":

        if noise_rate:
            train_dataset = PoisonNoiseCIFAR10(
                root=DataDir().get_dir("cifar-10"),
                train=True,
                transform=transform["train_transform"],
                download=True,
                poison_rate=poison_rate,
                noise_rate=noise_rate,
                noise_idx=noise_idx,
                perturbfile_path=perturbfile_path,
                hl_perturbfile_path=hl_perturbfile_path,
                perturb_type=perturb_type,
            )
        else:
            train_dataset = PoisonCIFAR10(
                root=DataDir().get_dir("cifar-10"),
                train=True,
                transform=transform["train_transform"],
                download=True,
                perturbfile_path=perturbfile_path
            )
    elif db_name=="cifar-100":
        if noise_rate:
            train_dataset = PoisonNoiseCIFAR100(
                root=DataDir().get_dir("cifar-100"),
                train=True,
                transform=transform["train_transform"],
                download=True,
                noise_rate=noise_rate,
                noise_idx=noise_idx,
                perturbfile_path=perturbfile_path,
                hl_perturbfile_path=hl_perturbfile_path,
                perturb_type=perturb_type,
            )
        else:
            train_dataset = PoisonCIFAR100(
                root=DataDir().get_dir("cifar-100"),
                train=True,
                transform=transform["train_transform"],
                download=True,
                perturbfile_path=perturbfile_path
            )

    elif db_name == "svhn":

        if noise_rate:
            train_dataset = PoisonNoiseSVHN(
                root=DataDir().get_dir("svhn"),
                split="train",
                transform=transform["train_transform"],
                download=True,
                noise_rate=noise_rate,
                noise_idx=noise_idx,
                perturbfile_path=perturbfile_path,
                hl_perturbfile_path=hl_perturbfile_path,
                perturb_type=perturb_type,
            )
        else:
            train_dataset = SVHN(
                root=DataDir().get_dir("svhn"),
                split="train",
                transform=transform["train_transform"],
                download=True
            )

    else:
        raise NotImplementedError(f"No {db_name} train dataset exist.")
    len_data = len(train_dataset)

    if perturbfile_path != None and hl_perturbfile_path == None:
        if noise_idx is None:
            return train_dataset
        subset_idx = torch.arange(len_data)[
            torch.where(torch.isin(torch.arange(len_data), noise_idx) == False, True, False)]
        # only train on clean samples + ue
        return Subset(train_dataset, subset_idx)
    elif perturbfile_path == None and hl_perturbfile_path == None and noise_idx is not None:
        subset_idx = torch.arange(len_data)[
            torch.where(torch.isin(torch.arange(len_data), noise_idx) == False, True, False)]
        return Subset(train_dataset, subset_idx)
    else:
        # train on clean samples + ue and noise samples + drm
        return train_dataset

def get_poison_dataset(db_name, transform, perturbfile_path):
    if db_name == "cifar-10":
        train_dataset = PoisonCIFAR10(
            root=DataDir().get_dir(db_name=db_name),
            train=True,
            transform=transform["train_transform"],
            download=True,
            perturbfile_path=perturbfile_path
        )
    elif db_name == "imagenet":

        train_dataset = PoisonImageNet(
            root=DataDir().get_dir("imagenet"),
            split="train",
            transform=transform["train_transform"],
            perturbfile_path=perturbfile_path
        )
    elif db_name == "tiny_imagenet":
        train_dataset = PoisonTinyImageNet(
            root=DataDir().get_dir(db_name),
            split="train",
            transform=transform["train_transform"],
            perturbfile_path=perturbfile_path

        )
    elif db_name == "imagenet_mini":
        train_dataset = PoisonImageNetMini(
            root=DataDir().get_dir(db_name),
            split="train",
            transform=transform["train_transform"],
        )

    elif db_name == "svhn":
        train_dataset = PoisonSVHN(
            root=DataDir().get_dir(db_name="svhn"),
            split="train",
            transform=transform["train_transform"],
            download=True,
            perturbfile_path=perturbfile_path
        )
    else:
        raise NotImplementedError(f"No {db_name} train dataset exist.")
    return train_dataset


def get_dataset(db_name, train_aug=True, perturbfile_path=None, ):
    # transform

    transform = get_transform(db_name)
    if not train_aug:
        transform["train_transform"] = transform["val_transform"]

    # dataset
    if not perturbfile_path:
        train_dataset = get_train_dataset(transform, db_name)
    else:
        train_dataset = get_poison_dataset(db_name, transform, perturbfile_path)
    val_dataset = get_val_dataset(db_name, transform)

    return {"train": train_dataset, "val": val_dataset}


def get_train_dataset(transform, db_name):
    if db_name == "cifar-10":
        train_dataset = CIFAR10(
            root=DataDir().get_dir(db_name=db_name),
            train=True,
            transform=transform["train_transform"],
            download=True,
        )
    elif db_name == "imagenet":

        train_dataset = ImageNet(
            root=DataDir().get_dir("imagenet"),
            split="train",
            transform=transform["train_transform"],
        )
    elif db_name == "tiny_imagenet":
        train_dataset = TinyImageNet(
            root=DataDir().get_dir(db_name),
            split="train",
            transform=transform["train_transform"],

        )
    elif db_name == "imagenet_mini":
        train_dataset = ImageNetMini(
            root=DataDir().get_dir(db_name),
            split="train",
            transform=transform["train_transform"],
        )

    elif db_name == "svhn":
        train_dataset = SVHN(
            root=DataDir().get_dir(db_name="svhn"),
            split="train",
            transform=transform["train_transform"],
            download=True
        )
    else:
        raise NotImplementedError(f"No {db_name} train dataset exist.")
    return train_dataset


def get_val_dataset(db_name, transform):
    if db_name == "cifar-10":
        val_dataset = CIFAR10(
            root=DataDir().get_dir(db_name),
            train=False,
            transform=transform["val_transform"],
            download=True
        )
    elif db_name =="cifar-100":
        val_dataset = CIFAR100(
            root=DataDir().get_dir(db_name),
            train=False,
            transform=transform["val_transform"],
            download=True
        )
    elif db_name == "imagenet":
        val_dataset = ImageNet(
            root=DataDir().get_dir(db_name),
            split="val",
            transform=transform["val_transform"],
        )
    elif db_name == "tiny_imagenet":
        val_dataset = TinyImageNet(
            root=DataDir().get_dir(db_name),
            split="val",
            transform=transform["val_transform"],
        )
    elif db_name == "imagenet_mini":
        val_dataset = ImageNetMini(
            root=DataDir().get_dir(db_name),
            split="val",
            transform=transform["val_transform"],

        )
    elif db_name == "svhn":
        val_dataset = SVHN(
            root=DataDir().get_dir("svhn"),
            split="test",
            transform=transform["val_transform"],
            download=True
        )
    else:
        raise NotImplementedError(f"No {db_name} val dataset exist.")
    return val_dataset


def get_transform(db_name,is_tensor=False):
    if db_name == "cifar-10" :
        if not is_tensor:
            return {
                "train_transform":
                    transforms.Compose([
                        # 此处不应为RandomResizedCrop
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]),
                "val_transform":
                    transforms.Compose([
                        transforms.ToTensor(),

                    ])
            }
        else:
            return {
                "train_transform":
                    transforms.Compose([
                        # 此处不应为RandomResizedCrop
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),

                    ]),
                "val_transform":
                    transforms.Compose([

                    ])
            }

    elif db_name=="cifar-100":

        return {
            "train_transform":
                transforms.Compose([
                    # 此处不应为RandomResizedCrop
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                ]),
            "val_transform":
                transforms.Compose([
                    transforms.ToTensor(),

                ])
        }
    elif db_name == "svhn":
        return {
            "train_transform": transforms.Compose([transforms.ToTensor()]),
            "val_transform": transforms.Compose([transforms.ToTensor()])
        }
    elif db_name in ["imagenet", "imagenet_mini"]:
        return {
            "train_transform":
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.4,
                                           hue=0.2),
                    transforms.ToTensor()
                ]),
            "val_transform": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        }
    elif db_name == "tiny_imagenet":
        return {
            "train_transform":
                transforms.Compose([
                    transforms.CenterCrop(256),
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()]),
            "val_transform":
                transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor()])
        }

    else:
        raise NotImplementedError("Invalid db_name. ")


if __name__ == '__main__':
    # test_get_transform()
    # test_get_criterion()
    pass
