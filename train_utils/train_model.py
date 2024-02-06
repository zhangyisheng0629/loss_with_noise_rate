#!/usr/bin/python
# author eson
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_utils.evaluator import Evaluator
from train_utils.logger import Logger
from train_utils.trainer import Trainer
from queue import Queue
device = "cuda" if torch.cuda.is_available() else "cpu"
def train_clean_model(ll_dataset, hl_loader,train_portion, clean_model: nn.Module, optimizer, criterion=nn.CrossEntropyLoss(),
                      train_epochs=5, train_on="noise"):
    train_size = int(train_portion * len(ll_dataset))
    val_size = len(ll_dataset) - train_size
    ll_train,ll_val = torch.utils.data.random_split(ll_dataset, [train_size, val_size])
    ll_train_loader = DataLoader(ll_train, 128, True, drop_last=True, num_workers=4)
    ll_val_loader = DataLoader(ll_val, 128, False, drop_last=False, num_workers=4)
    trainer = Trainer(
        logger=Logger(start_epoch=0, log_loss=False),
        dataloader=ll_train_loader,
        model=clean_model,
        optimizer=optimizer,
        criterion=criterion,
        cur_epoch=0,
        total_epoch=train_epochs,
        device=device
    )
    evaluator = Evaluator(
        logger=Logger(start_epoch=0,log_loss=False),
        dataloader=ll_val_loader,
        model=clean_model,
        criterion=criterion,
        loss_logger=None,
        cur_epoch=0,
        total_epoch=train_epochs,
        device=device

    )

    val_evaluator=Evaluator(
        logger=Logger(start_epoch=0, log_loss=False),
        dataloader=hl_loader,
        model=clean_model,
        criterion=criterion,
        loss_logger=None,
        cur_epoch=0,
        total_epoch=train_epochs,
        device=device
    )

    early_list=[]
    train_acc=0.0
    val_acc=0.0
    while val_acc<0.9:

        new_train_acc=trainer.train(train_on)
        evaluator.eval(eval_on="noise")
        print("Eval on high loss samples. ")
        val_acc=val_evaluator.eval(eval_on="clean")
        if early_list and new_train_acc>early_list[0]:
            early_list=[new_train_acc]

        else:
            early_list.append(new_train_acc)
        print("最近四次acc: ",[round(i,2) for i in  early_list])
        if len(early_list)>=4:
            return clean_model

        trainer.logger.new_epoch()
        evaluator.logger.new_epoch()
        val_evaluator.logger.new_epoch()
    return clean_model