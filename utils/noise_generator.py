#!/usr/bin/python
# author eson
import time

import numpy as np
import torch
import torchvision.models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchattacks import PGD
from tqdm import tqdm
from torchattacks.attack import Attack

from train.logger import Logger
from utils.get import get_dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class NoiseGenerator():
    def __init__(self, loader, order_train_loader, hl_loader, model, train_steps=10,
                 attack: Attack = PGD,
                 drm_attack: Attack = None):
        self.loader = loader
        self.model = model
        self.train_steps = train_steps
        self.device = device
        self.attack = attack
        self.epsilon = attack.eps
        self.step_size = attack.steps
        self.noise_len = len(order_train_loader.dataset)
        # attack target: original label
        # self.attack.set_mode_targeted_by_function(target_map_function=lambda images, labels: labels)
        self.noise = torch.zeros([self.noise_len, 3, 32, 32]).cuda()
        # self.noise = self.random_noise([50000, 3, 32, 32])
        self.data_iter = iter(self.loader)
        self.order_train_loader = order_train_loader

        self.hl_noise = torch.zeros([self.noise_len, 3, 32, 32]).cuda()
        self.drm_attack = drm_attack
        self.hl_loader = hl_loader

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

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        """
        PGD method to generate noise_img
        Args:
            images:
            labels:
            model:
            optimizer:
            criterion:
            random_noise:
            sample_wise:

        Returns:

        """

        adv_images = images + noise
        adv_images = adv_images.clone().detach()

        eta = random_noise
        for _ in range(self.step_size):
            adv_images.requires_grd = True
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            # perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(eta, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def outer_optim(self, optimizer, criterion, pos):
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
                self.data_iter = iter(self.data_iter)
                batch = next(self.data_iter)
            if "Noise" in self.loader.dataset.__class__.__name__ or self.loader.dataset.__class__.__name__ == "Subset":
                (X, noise_target, true_target, if_noise) = batch.values()
                X, y = X, noise_target
            else:
                X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            # print(y)
            # add noise
            print(pos)
            X, pos = self.add_noise(X, pos)
            # bug
            # print(self.model(X))
            # train model

            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            if criterion.reduction == "none":
                loss = loss.mean()
            loss.backward()
            optimizer.step()

        return pos

    def inner_optim(self, logger: Logger):
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
            if "Noise" in self.loader.dataset.__class__.__name__ or self.loader.dataset.__class__.__name__ == "Subset":

                (X, noise_target, true_target, if_noise) = batch.values()
                X, y = X, noise_target
            else:
                X, y = batch
            batch_num = X.shape[0]

            # train on images + noise
            images, labels = X.to(self.device), y.to(self.device)
            noise = self.noise[pos:pos + batch_num]
            if self.attack.__class__.__name__=="PGD":
                adv_images = self.attack(images, labels)
            elif self.attack.__class__.__name__=="UEPGD":
                adv_images = self.attack(images, labels, noise)
            else:
                raise ValueError
            # eval noise

            output = self.model(images)
            adv_output = self.model(adv_images)

            n1 = torch.sum(
                torch.eq(
                    torch.argmax(output, 1), labels
                )
            )

            pred = torch.argmax(adv_output, 1)
            correct_num = torch.sum(torch.eq(pred, labels))

            payload = {
                "correct_num": correct_num.item(),
                "batch_num": batch_num
            }
            logger.log(payload)

            t_loader.set_postfix(postfix=f"cur_pos|total_pos:{pos}|{len(self.loader.dataset)},"
                                         f"correct num {n1}---->{correct_num},acc: {logger.cal_acc()}")
            pos = self.update_noise(pos, adv_images - images)
        accuracy = logger.cal_acc()
        return accuracy

    def ue(self, optimizer, criterion, stop_error, logger):
        SUCCESS = False
        train_pos = 0
        while not SUCCESS:
            # repeat to find error minimizing perturbation
            logger.new_epoch()
            train_pos = self.outer_optim(optimizer, criterion, train_pos)

            accuracy = self.inner_optim(logger)

            SUCCESS = (1 - accuracy) < stop_error

        return self.noise

    def deep_representation_manipulation(self, criterion):
        print("deep_representation_manipulation")

        t_loader = tqdm(enumerate(self.hl_loader), total=len(self.hl_loader))
        pos = 0
        for i, batch in t_loader:
            if "Noise" in self.loader.dataset.__class__.__name__ or self.loader.dataset.__class__.__name__ == "Subset":

                (X, noise_target, true_target, if_noise) = batch.values()
                X, y = X, noise_target
            else:
                X, y = batch
            batch_num = X.shape[0]

            # train on images + noise
            images = X.to(self.device)
            adv_images = self.drm_attack(images)
            pos = self.update_noise(pos, adv_images - images)
            t_loader.set_postfix(postfix=f"cur_pos|total_pos:{pos}|{len(self.hl_loader.dataset)},")

        return self.hl_noise


if __name__ == '__main__':
    dataset = get_dataset(
        "ue_gen",
        "cifar-10",
        noise_rate=0,
        train_aug=False
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
