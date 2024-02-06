#!/usr/bin/python
# author eson
from typing import List
from argparse import ArgumentParser
import torch
from torch.optim import Optimizer, Adam
import torch.nn as nn
from tqdm import tqdm, trange

from train_utils.logger import Logger
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

    def train(self, train_on="clean"):
        t_loader = tqdm(enumerate(self.dataloader))
        for i, batch in t_loader:
            payload = self.train_batch(i, batch, train_on, log_loss=False)

            self.logger.log(payload)
            t_loader.set_description(f"Epoch {self.cur_epoch} batch {i}/{len(self.dataloader)}")
            postfix = self.get_postfix()
            t_loader.set_postfix(postfix=postfix)

        self.cur_epoch += 1
        return self.logger.cal_acc()

    def get_postfix(self, ):
        return f"accuracy {sum(self.logger.correct_num) / self.logger.total_num:.2f}"

    def train_batch(self, i, batch, train_on, log_loss):
        self.model.train()
        if len(batch) == 4:
            # X, noise_target, true_target, if_noise,item

            if train_on == "noise":
                X = batch["image"].to(self.device, non_blocking=True)
                y = batch["noise_target"].to(self.device, non_blocking=True)
                if_noise = batch["if_noise"].to(self.device, non_blocking=True)
                # item = batch["item"].to(self.device, non_blocking=True)

            elif train_on == "clean":
                X = batch["image"].to(self.device, non_blocking=True)
                y = batch["true_target"].to(self.device, non_blocking=True)
            else:
                raise ValueError(f"Invalid train_on {train_on}")
        elif len(batch) == 2:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
        else:
            raise ValueError("Illegal batch.")
        self.optimizer.zero_grad()
        output = self.model(X)
        if type(self.criterion).__name__ == "TruncatedLoss":
            loss = self.criterion(output, y, item)
        else:
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
                payload[k] = v.item()

        loss = torch.mean(loss) if len(loss.shape) >= 1 else loss

        loss.backward()
        self.optimizer.step()
        return payload
