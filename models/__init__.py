#!/usr/bin/python
# author eson
import mlconfig
import torch
import torch.nn as nn
import torchvision


from . import ResNet

mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
mlconfig.register(torch.nn.CrossEntropyLoss)


# Models

mlconfig.register(ResNet.ResNet18)

mlconfig.register(ResNet.ResNet34)
mlconfig.register(ResNet.ResNet50)
mlconfig.register(ResNet.ResNet101)
mlconfig.register(ResNet.ResNet152)
# mlconfig.register(ToyModel.ToyModel)
# mlconfig.register(DenseNet.DenseNet121)
# mlconfig.register(inception_resnet_v1.InceptionResnetV1)

# torchvision models
mlconfig.register(torchvision.models.resnet18)
mlconfig.register(torchvision.models.resnet50)
mlconfig.register(torchvision.models.densenet121)


