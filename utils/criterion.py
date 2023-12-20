#!/usr/bin/python
# author eson
import torch.nn as nn

def get_criterion(reduction):
    c=nn.CrossEntropyLoss(reduction=reduction)
    return c



if __name__ == '__main__':
    kwargs={"reduction":"none"}
    c=get_criterion(**kwargs)
    pass