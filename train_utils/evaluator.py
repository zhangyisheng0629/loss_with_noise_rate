#!/usr/bin/python
# author eson
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_utils.logger import Logger
from train_utils.loss_logger import LossLogger

from utils.common_utils import ConfusionMatrixDrawer


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

    def eval(self,cmd=None,eval_on="clean",log_loss=False):
        t_loader = tqdm(enumerate(self.dataloader))
        for i, batch in t_loader:
            payload = self.eval_batch(i, batch,cmd,eval_on=eval_on,log_loss=log_loss)
            self.logger.log(payload)
            t_loader.set_description(f"Epoch {self.cur_epoch} batch {i}/{len(self.dataloader)}")
            t_loader.set_postfix(postfix=self.get_postfix())
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
            return self.logger.cal_acc()

    def get_postfix(self, ):

        return f"accuracy {sum(self.logger.correct_num) / self.logger.total_num:.2f}"

    def recog(self):
        return self.loss_logger.recog_noise()

    def eval_batch(self, i, batch,cmd:ConfusionMatrixDrawer=None,eval_on="clean",log_loss=False):
        self.model.eval()
        if len(batch)==4:
            if eval_on == "clean":
                X = batch["image"]
                y = batch["true_target"]

            elif eval_on == "noise":
                X = batch["image"]
                y = batch["noise_target"]
                if_noise = batch["if_noise"]
            else:
                raise ValueError
        elif len(batch)==2:
            (X, y) = batch
        else:
            raise ValueError("Illegal batch")
        X, y = X.to(self.device), y.to(self.device)
        output = self.model(X)

        if self.loss_logger:
            loss = self.criterion(output, y)
            self.loss_logger.update(loss)
        pred = torch.argmax(output, dim=1)
        batch_num = len(y)
        correct_num = torch.sum(torch.eq(pred, y))

        if cmd:
            cmd.update(y,pred)
        payload = {
            "batch_num": batch_num,
            "correct_num": correct_num,
        }

        if log_loss:
            if self.criterion.__class__.__name__ == "CrossEntropyLoss":
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
        return payload
