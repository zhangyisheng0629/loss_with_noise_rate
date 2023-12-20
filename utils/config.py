#!/usr/bin/python
# author eson


import mlconfig
from mlconfig import instantiate

def load_conf(file_path):
    config = mlconfig.load(file_path)
    return config


if __name__ == '__main__':
    file_path = "../configs/resnet18_cifar10.yaml"
    conf = load_conf(file_path)
    instantiate(conf.model)
