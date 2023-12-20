#!/usr/bin/python
# author eson
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.logger import Logger
from train.loss_logger import LossLogger


class Evaluator:
    def __init__(self,
                 logger: Logger,
                 dataloader: DataLoader,
                 model: nn.Module,
                 criterion: Optional[nn.Module] = nn.CrossEntropyLoss(),
                 loss_logger: Optional[LossLogger] = None,
                 args=None,
                 cur_epoch=0,
                 total_epoch=0,
                 device: str = "cpu",

                 ):
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.args = args
        self.cur_epoch = cur_epoch
        self.total_epoch = total_epoch
        self.device = device
        self.loss_logger = loss_logger
        self.noise_precision = 0
        self.noise_recall = 0
        self.recog_noise = True if self.loss_logger else False

    def eval(self, ):
        t_loader = tqdm(enumerate(self.dataloader))
        for i, batch in t_loader:
            payload = self.eval_batch(i, batch)
            self.logger.log(payload)

            # TODO : fun get_postfix() to get the postfix string show in the process bar
            # postfix=
            t_loader.set_description(f"Epoch {self.cur_epoch} batch {i}/{len(self.dataloader)}")
            t_loader.set_postfix(
                postfix=f"accuracy {sum(self.logger.correct_num) / self.logger.total_num:.2f}, "

            )
        # update recog noise rate per 100 steps
        if self.recog_noise:
            # TODO：precision and recall
            self.noise_precision, self.noise_recall = self.recog()
            print(f"Epoch {self.cur_epoch}, "
                  f"noise precision {self.noise_precision:.2f} %, "
                  f"noise recall {self.noise_recall:.2f} % ")
        self.cur_epoch += 1
        if self.recog_noise:
            return {"precision": self.noise_precision,
                    "recall": self.noise_recall}
        else:
            return {}

    def recog(self):
        return self.loss_logger.recog_noise()

    def eval_batch(self, i, batch):
        self.model.eval()
        if "Noise" in self.dataloader.dataset.__class__.__name__:
            X, y = batch["image"], batch["noise_target"]
        # subset
        elif self.dataloader.dataset.__class__.__name__ == "Subset" and "Noise" in self.dataloader.dataset.dataset.__class__.__name__:
            X, y = batch["image"], batch["noise_target"]
        else:
            (X, y) = batch
        X, y = X.to(self.device), y.to(self.device)
        # print(batch["true_target"])
        output = self.model(X)

        if self.criterion and self.loss_logger:
            loss = self.criterion(output, y)
            self.loss_logger.update(loss)
        pred = torch.argmax(output, dim=1)
        batch_num = len(y)
        correct_num = torch.sum(torch.eq(pred, y))

        payload = {
            "batch_num": batch_num,
            "correct_num": correct_num,
        }

        for k, v in payload.items():
            if torch.is_tensor(v):
                payload[k] = round(v.item(), 2)

        # "acc_avg": self.acc_meters.avg,
        # "loss": loss,
        # "loss_avg": self.loss_meters.avg,
        # "lr": self.optimizer.param_groups[0]['lr'],
        # "|gn|": grad_norm

        return payload
