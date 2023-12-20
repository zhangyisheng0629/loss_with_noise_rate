#!/usr/bin/python
# author eson
# !/usr/bin/python
# author eson
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, SVHN

from data.cifar import CIFAR10, PoisonNoiseCIFAR10, PoisonCIFAR10
from data.data_dir import DataDir
from data.imagenet import NoiseImageNet
from data.noise_cifar import NoiseCIFAR10
from data.svhn import NoiseSVHN
from data.tiny_imagenet import NoiseTinyImageNet, TinyImageNet
from data.imagenet_mini import ImageNetMini, NoiseImageNetMini


def get_dataset(stage, db_name, noise_rate=0, train_aug=True, noise_idx=None,perturbfile_path=None):
    # transform
    transform = get_transform(db_name)
    if not train_aug:
        transform["train_transform"] = transform["val_transform"]

    # dataset
    if stage == "select":
        train_dataset = get_train_dataset_stage12(transform, db_name, noise_rate, )
    elif stage == "ue_gen":
        train_dataset = get_train_dataset_stage12(transform, db_name, noise_rate, noise_idx=noise_idx)
    elif stage == "ue_train":
        train_dataset = get_train_dataset_stage3(transform, db_name, noise_rate, noise_idx=noise_idx,
                                                 perturbfile_path=perturbfile_path)

    else:
        raise ValueError(f"Invalid stage name {stage}, set it from 'select, ue_gen, ue_train'. ")

    val_dataset = get_val_dataset(db_name, transform)

    return {"train": train_dataset, "val": val_dataset}


def get_train_dataset_stage12(transform, db_name, noise_rate: float = 0, noise_idx=None):
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
                root=DataDir().get_dir(db_name=db_name),
                train=True,
                transform=transform["train_transform"],
                download=True,

            )

    elif db_name == "imagenet":

        if noise_rate:
            train_dataset = NoiseImageNet(
                root=DataDir().get_dir("imagenet"),
                split="train",
                transform=transform["train_transform"],
                noise_rate=noise_rate
            )
        else:
            train_dataset = ImageNet(
                root=DataDir().get_dir("imagenet"),
                split="train",
                transform=transform["train_transform"],

            )

    elif db_name == "tiny_imagenet":
        if noise_rate:
            train_dataset = NoiseTinyImageNet(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],
                noise_rate=noise_rate
            )
        else:
            train_dataset = TinyImageNet(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],

            )

    elif db_name == "imagenet_mini":
        if noise_rate:
            train_dataset = NoiseImageNetMini(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],
                noise_rate=noise_rate
            )
        else:
            train_dataset = ImageNetMini(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],
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
                root=DataDir().get_dir(db_name="svhn"),
                split="train",
                transform=transform["train_transform"],
                download=True
            )

    else:
        raise NotImplementedError(f"No {db_name} train dataset exist.")
    if noise_idx!=None:
        subset_idx = torch.arange(50000)[torch.where(torch.isin(torch.arange(50000), noise_idx) == False, True, False)]
        train_dataset = get_subset(train_dataset, subset_idx)

    return train_dataset


def get_train_dataset_stage3(transform, db_name, noise_rate: float = 0, noise_idx=None, perturbfile_path=None):
    if db_name == "cifar-10":

        if noise_rate:
            train_dataset = PoisonNoiseCIFAR10(
                root=DataDir().get_dir("cifar-10"),
                train=True,
                transform=transform["train_transform"],
                download=True,
                noise_rate=noise_rate,
                noise_idx=noise_idx,
                perturbfile_path=perturbfile_path,
            )
        else:
            train_dataset = PoisonCIFAR10(
                root=DataDir().get_dir(db_name=db_name),
                train=True,
                transform=transform["train_transform"],
                download=True,
                perturbfile_path=perturbfile_path

            )



    elif db_name == "imagenet":

        if noise_rate:
            train_dataset = PoisonNoiseImageNet(
                root=DataDir().get_dir("imagenet"),
                split="train",
                transform=transform["train_transform"],
                noise_rate=noise_rate
            )
        else:
            train_dataset = ImageNet(
                root=DataDir().get_dir("imagenet"),
                split="train",
                transform=transform["train_transform"],

            )

    elif db_name == "tiny_imagenet":
        if noise_rate:
            train_dataset = PoisonNoiseTinyImageNet(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],
                noise_rate=noise_rate
            )
        else:
            train_dataset = TinyImageNet(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],

            )

    elif db_name == "imagenet_mini":
        if noise_rate:
            train_dataset = PoisonNoiseImageNetMini(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],
                noise_rate=noise_rate
            )
        else:
            train_dataset = ImageNetMini(
                root=DataDir().get_dir(db_name),
                split="train",
                transform=transform["train_transform"],
            )

    elif db_name == "svhn":

        if noise_rate:
            train_dataset = PoisonNoiseSVHN(
                root=DataDir().get_dir("svhn"),
                split="train",
                transform=transform["train_transform"],
                download=True,
                noise_rate=noise_rate
            )
        else:
            train_dataset = SVHN(
                root=DataDir().get_dir(db_name="svhn"),
                split="train",
                transform=transform["train_transform"],
                download=True
            )

    else:
        raise NotImplementedError(f"No {db_name} train dataset exist.")
    if noise_idx!=None:
        subset_idx = torch.arange(50000)[torch.where(torch.isin(torch.arange(50000), noise_idx) == False, True, False)]
        train_dataset = get_subset(train_dataset, subset_idx)
    return train_dataset


def get_val_dataset(db_name, transform):
    if db_name == "cifar-10":
        val_dataset = CIFAR10(
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


def get_subset(dataset, idx):
    return Subset(dataset, idx)


def get_transform(db_name):
    if db_name == "cifar-10":
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

                    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    #                      std=[0.2023, 0.1994, 0.2010])
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
