#!/usr/bin/python
# author eson
import torchvision.transforms as transforms


def get_transform(db_name):
    if db_name == "cifar-10":
        return {
            "train_transform":
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    #                      std=[0.2023, 0.1994, 0.2010])
                ]),
            "val_transform":
                transforms.Compose([
                    transforms.ToTensor(),

                    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    #                      std=[0.2023, 0.1994, 0.2010])
                ])
        }

    else:
        raise NotImplementedError
