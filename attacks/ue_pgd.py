#!/usr/bin/python
# author eson

import torch
from torch import nn
from torchattacks.attack import Attack
from torchattacks import PGD


class UEPGD(PGD):
    def __init__(self, model, eps, alpha, steps):
        super().__init__(model, eps, alpha, steps)
        self.random_start = False

    def forward(self, images, labels, noise):
        r"""
                Overridden.
                """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images + noise
        adv_images = adv_images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):

            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class EOTUEPGD(PGD):
    def __init__(self, model, eps, alpha, steps, trans, sample_num=5):
        super().__init__(model, eps, alpha, steps)
        self.random_start = False
        self.trans = trans
        self.sample_num = sample_num

    def forward(self, images, labels, noise):
        r"""
                Overridden.
                """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images + noise
        adv_images = adv_images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # for _ in range(self.steps):
        #     adv_images.requires_grad = True
        #     trans_adv_images = self.trans(adv_images)
        #     outputs = self.get_logits(trans_adv_images)
        #
        #     # Calculate loss
        #     if self.targeted:
        #         cost = -loss(outputs, target_labels)
        #     else:
        #         cost = loss(outputs, labels)
        #
        #     # Update adversarial images
        #     grad = torch.autograd.grad(
        #         cost, adv_images, retain_graph=False, create_graph=False
        #     )[0]
        #
        #     adv_images = adv_images.detach() + self.alpha * grad.sign()
        #     delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        #     adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        #
        # return adv_images

        for _ in range(self.steps):
            adv_images.requires_grad = True

            sum_grad=torch.zeros_like(adv_images)
            for _ in range(self.sample_num):
                trans_adv_images = self.trans(adv_images)
                outputs = self.get_logits(trans_adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                sum_grad+=grad

            adv_images = adv_images.detach() + self.alpha * sum_grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
