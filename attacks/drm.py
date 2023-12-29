#!/usr/bin/python
# author eson
import torch
from torchattacks.attack import Attack


class DeepReprentationAttack(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=20, criterion=None):
        super().__init__("DeepReprentationAttack", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.criterion = criterion

    def forward(self, images, labels=None):
        logits = self.model(images)

        perturb_img = images.clone().detach()
        for _ in range(self.steps):
            perturb_img.requires_grad = True
            outputs = self.model(perturb_img)

            # far from the deep representation
            cost = -self.criterion(outputs, logits)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, perturb_img, retain_graph=False, create_graph=False
            )[0]

            perturb_img = perturb_img.detach() + self.alpha * grad.sign()
            delta = torch.clamp(perturb_img - images, min=-self.eps, max=self.eps)
            perturb_img = torch.clamp(images + delta, min=0, max=1).detach()

        return perturb_img


