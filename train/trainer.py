#!/usr/bin/python
# author eson
from typing import List
from argparse import ArgumentParser
import torch
from torch.optim import Optimizer, Adam
import torch.nn as nn
from tqdm import tqdm, trange

from train.logger import Logger
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self,
                 logger: Logger,
                 dataloader: DataLoader,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 args=None,
                 cur_epoch=0,
                 total_epoch=0,
                 device: str = "cpu",

                 ):
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.cur_epoch = cur_epoch
        self.total_epoch = total_epoch
        self.device = device

    def train(self, train_on="noise", log_loss=False):
        t_loader = tqdm(enumerate(self.dataloader))
        for i, batch in t_loader:
            payload = self.train_batch(i, batch, train_on, log_loss)

            self.logger.log(payload)
            t_loader.set_description(f"Epoch {self.cur_epoch} batch {i}/{len(self.dataloader)}")
            postfix = self.get_postfix(log_loss)
            t_loader.set_postfix(postfix=postfix)

        self.cur_epoch += 1
        return self.logger

    def get_postfix(self, log_loss=False):
        if log_loss:
            return (f"clean_avg_loss {self.logger.cal_avg_loss('clean'):.2f} , "
                    f"noise_avg_loss {self.logger.cal_avg_loss('noise'):.2f} , "
                    f"accuracy {sum(self.logger.correct_num) / self.logger.total_num:.2f}")
        else:
            return f"accuracy {sum(self.logger.correct_num) / self.logger.total_num:.2f}"

    def train_batch(self, i, batch, train_on, log_loss=False):
        self.model.train()
        if "Noise" in self.dataloader.dataset.__class__.__name__ or \
                self.dataloader.dataset.__class__.__name__ == "Subset":
            (X, noise_target, true_target, if_noise) = batch.values()
            if train_on == "noise":
                X, y = X.to(self.device, non_blocking=True), noise_target.to(self.device, non_blocking=True)
            elif train_on == "clean":
                X, y = X.to(self.device, non_blocking=True), true_target.to(self.device, non_blocking=True)
            else:
                raise ValueError(f"Invalid train_on {train_on}")
        else:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)

        batch_num = X.shape[0]
        pred = torch.argmax(output, dim=1)
        correct_num = torch.sum(torch.eq(pred, y))
        payload = {"correct_num": correct_num,
                   "batch_num": batch_num, }

        if log_loss:
            assert self.criterion.reduction == "none"
            batch_loss = torch.sum(loss)
            clean_loss = torch.sum(loss[if_noise == False])
            noise_loss = torch.sum(loss[if_noise == True])

            noise_num = torch.sum(if_noise).item()
            clean_num = (if_noise == False).sum().item()
            """
                batch_noise_number:批量中的噪声个数
                batch_loss：批量损失        
            """
            payload.update({
                "clean_loss": clean_loss,
                "noise_loss": noise_loss,
                "clean_num": clean_num,
                "noise_num": noise_num,
                "batch_loss": batch_loss,
            })
            for k, v in payload.items():
                if torch.is_tensor(v):
                    payload[k] = round(v.item(), 2)

        if self.criterion.reduction == "none":
            loss = torch.mean(loss)

        loss.backward()
        self.optimizer.step()
        return payload
