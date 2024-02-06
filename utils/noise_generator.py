#!/usr/bin/python
# author eson
import os
import time
from typing import List

import numpy as np
import torch
import torchvision.models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchattacks import PGD
from tqdm import tqdm
from torchattacks.attack import Attack

from train_utils.logger import Logger
from utils.common_utils import ConfusionMatrixDrawer
from utils.criterion import MixCrossEntropyLoss, SCANLoss
from utils.get import get_dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# C=SCANLoss(entropy_weight=6.0)
class NoiseGenerator():
    def __init__(self, outer_train_loader, order_train_loader, model, train_steps=10,
                 attack: PGD = None):
        self.outer_train_loader = outer_train_loader
        self.data_iter = iter(self.outer_train_loader)
        self.order_train_loader = order_train_loader

        self.model = model
        self.train_steps = train_steps
        self.device = device
        self.attack = attack
        self.epsilon = attack.eps
        self.step_size = attack.steps

        self.noise_len = len(order_train_loader.dataset)
        self.noise = torch.zeros([self.noise_len, 3, 32, 32]).cuda()

    def random_noise(self, noise_shape=[50000, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        # random_noise=torch.zeros(noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise.to(self.device)

    def update_noise(self, pos, batch_noise: torch.Tensor):
        n = batch_noise.shape[0]
        self.noise[pos:pos + n] = batch_noise
        return pos + n

    def add_noise(self, X: torch.Tensor, pos):
        """

        Args:
            X: 原图片
            pos: 加噪声的位置

        Returns:

        """
        for i, image in enumerate(X):
            X[i] += self.noise[pos + i]
        return X, pos + X.shape[0]

    def outer_optim(self, *args, **kwargs):
        raise NotImplementedError("Outer_optim fun not be overwitten. ")

    def inner_optim_samplewise(self, *args, **kwargs):
        raise NotImplementedError("Inner_optim fun not be overwitten. ")

    def ue(self, *args, **kwargs):
        raise NotImplementedError("UE fun not be overwitten. ")


class UENoiseGenerator(NoiseGenerator):
    def __init__(self, outer_train_loader, order_train_loader, model=None, train_steps=10,
                 attack: PGD = None):
        super().__init__(outer_train_loader, order_train_loader, model, train_steps, attack)

    def outer_optim(self, optimizer, criterion, pos, gen_on,trans):
        """
        solve optimization problem for updating $\theta$
        Returns:

        """
        self.attack.set_model_training_mode(model_training=True)
        self.model.train()
        # pos is the current train position
        print(f"Train {self.train_steps} steps.")

        for step in tqdm(range(self.train_steps)):
            try:
                batch = next(self.data_iter)
            except Exception as e:
                self.data_iter = iter(self.outer_train_loader)
                batch = next(self.data_iter)
                pos = 0
            if len(batch) == 2:
                X, y = batch
            elif len(batch) == 4:
                if gen_on == "clean":
                    X, _, y, _ = batch.values()
                elif gen_on == "noise":
                    X, y, _, _ = batch.values()
                else:
                    raise ValueError
            X, y = X.to(self.device), y.to(self.device)
            X_noise = X.clone().detach()
            # add noise
            X_noise, pos = self.add_noise(X_noise, pos)
            trans_X_noise=trans(X_noise) if trans else X_noise
            optimizer.zero_grad()
            output = self.model(trans_X_noise)

            loss = criterion(output, y)

            loss = loss.mean() if len(loss.shape) >= 1 else loss
            loss.backward()
            optimizer.step()

        return pos

    def inner_optim_samplewise(self, logger, gen_on):
        """
        pgd method
        Returns:
        for step in self.train_steps:
        """
        self.attack.set_model_training_mode(model_training=False)
        self.model.eval()
        pos = 0
        t_loader = tqdm(enumerate(self.order_train_loader), total=len(self.order_train_loader), ncols=150)
        for i, batch in t_loader:
            if len(batch) == 2:
                X, y = batch
            elif len(batch) == 4:
                if gen_on == "clean":
                    X, _, y, _ = batch.values()
                elif gen_on == "noise":
                    X, y, _, _ = batch.values()
                else:
                    raise ValueError
            batch_num = X.shape[0]

            # train on images + noise
            images, labels = X.to(self.device), y.to(self.device)
            noise = self.noise[pos:pos + batch_num]
            if self.attack.__class__.__name__ == "PGD":
                adv_images = self.attack(images, labels)
            elif self.attack.__class__.__name__ in ["UEPGD", "EOTUEPGD"]:
                adv_images = self.attack(images, labels, noise)

            else:
                raise ValueError
            # eval noise
            output = self.model(images)
            adv_output = self.model(adv_images)

            origin = torch.sum(torch.eq(torch.argmax(output, 1), labels))
            pred = torch.argmax(adv_output, 1)
            correct_num = torch.sum(torch.eq(pred, labels))
            payload = {
                "correct_num": correct_num.item(),
                "batch_num": batch_num
            }
            logger.log(payload)
            t_loader.set_postfix(postfix=f"cur_pos|total_pos:{pos}|{len(self.outer_train_loader.dataset)}, "
                                         f"correct {origin}---->{correct_num}, "
                                         f"acc: {logger.cal_acc():.2f}")
            pos = self.update_noise(pos, adv_images - images)

        return logger.cal_acc()

    def immer_optim_classwise(self):

        pass

    def ue(self, optimizer, criterion, stop_error, logger: Logger, draw_confusion=True,
           val_loader=None, gen_on="clean", num_classes=10,trans=None,res_folder=None):
        SUCCESS = False
        train_pos = 0
        iterative_num=0
        if draw_confusion:
            assert val_loader is not None
            cmd = ConfusionMatrixDrawer(num_classes)
        while not SUCCESS:
            # repeat to find error minimizing perturbation
            iterative_num+=1
            print("Iterative num {}".format(iterative_num))
            logger.new_epoch()
            train_pos = self.outer_optim(optimizer, criterion, train_pos, gen_on,trans)  # train model for {steps} steps

            acc = self.inner_optim_samplewise(logger, gen_on)

            SUCCESS = (1 - acc) < stop_error
            if draw_confusion:
                cmd.reset()
                cmd.update_draw(val_loader, self.model, self.device, title="UE")
                print(cmd.confusion)

            # save
            if res_folder:
                print(f"Perturbation saved at {os.path.join(res_folder, 'perturbation.pt')}. ")
                torch.save(self.noise, os.path.join(res_folder, "perturbation.pt"))
        return self.noise

    def deep_representation_manipulation(self, hl_loader, hl_attack):

        self.hl_loader = hl_loader
        self.hl_noise_len = len(hl_loader.dataset)
        self.hl_noise = torch.zeros([self.hl_noise_len, 3, 32, 32]).cuda()
        print("deep_representation_manipulation")

        t_loader = tqdm(enumerate(self.hl_loader), total=len(self.hl_loader))
        pos = 0
        for i, batch in t_loader:
            if "Noise" in self.hl_loader.dataset.__class__.__name__ or self.hl_loader.dataset.__class__.__name__ == "Subset":

                (X, noise_target, true_target, if_noise) = batch.values()
                X, y = X, noise_target
            else:
                X, y = batch
            images, labels = X.to(self.device), y.to(self.device)
            adv_images = hl_attack(images, labels)
            pos = self.update_noise(pos, adv_images - images)
            t_loader.set_postfix(postfix=f"cur_pos|total_pos:{pos}|{len(self.hl_loader.dataset)},")

        return self.hl_noise


if __name__ == '__main__':
    dataset = get_dataset(
        "ue_gen",
        "cifar-10",
    )
    train_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    order_train_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    logger = Logger(0, log_loss=False)
    model = torchvision.models.resnet18().to(device)
    model.eval()
    attack = PGD(model, eps=8 / 255, steps=20, alpha=0.8 / 255)
    attack.set_mode_targeted_by_function(target_map_function=lambda images, labels: labels)
    ng = NoiseGenerator(train_loader, order_train_loader, model, 10, attack)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4,
                                momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    noise = ng.ue(optimizer, criterion, 0.01, logger)
