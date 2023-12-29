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

    def eval(self, eval_on="noise", log_loss=False):
        t_loader = tqdm(enumerate(self.dataloader))
        for i, batch in t_loader:
            payload = self.eval_batch(i, batch, eval_on, log_loss)
            self.logger.log(payload)

            t_loader.set_description(f"Epoch {self.cur_epoch} batch {i}/{len(self.dataloader)}")
            t_loader.set_postfix(postfix=self.get_postfix(log_loss))
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

    def get_postfix(self, log_loss=False):
        if log_loss:
            return (f"clean_avg_loss {self.logger.cal_avg_loss('clean'):.2f} , "
                    f"noise_avg_loss {self.logger.cal_avg_loss('noise'):.2f} , "
                    f"accuracy {sum(self.logger.correct_num) / self.logger.total_num:.2f}")
        else:
            return f"accuracy {sum(self.logger.correct_num) / self.logger.total_num:.2f}"

    def recog(self):
        return self.loss_logger.recog_noise()

    def eval_batch(self, i, batch, eval_on, log_loss):
        self.model.eval()
        if "Noise" in self.dataloader.dataset.__class__.__name__ or (
                self.dataloader.dataset.__class__.__name__ == "Subset" and "Noise" in self.dataloader.dataset.dataset.__class__.__name__):
            (X, noise_target, true_target, if_noise) = batch.values()
            if eval_on == "clean":
                X, y = X, true_target
            elif eval_on == "noise":
                X, y = X, noise_target
            else:
                raise ValueError
        else:
            (X, y) = batch
        X, y = X.to(self.device), y.to(self.device)
        output = self.model(X)

        if  self.loss_logger:
            loss = self.criterion(output, y)
            self.loss_logger.update(loss)
        pred = torch.argmax(output, dim=1)
        batch_num = len(y)
        correct_num = torch.sum(torch.eq(pred, y))

        payload = {
            "batch_num": batch_num,
            "correct_num": correct_num,
        }

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
        return payload
