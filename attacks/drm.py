#!/usr/bin/python
# author eson
import torch
from torchattacks.attack import Attack


class DeepRepresentationAttack(object):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=20, criterion=None,trans=None,target_map=None,target=True):
        self.model=model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.criterion = criterion
        self.target_map=target_map
        self.device = next(model.parameters()).device
        self.target=target
        self.trans=trans
    def __call__(self, images, labels=None):
        target=self.target_map(images,labels).to(self.device)

        perturb_img = images.clone().detach()
        for _ in range(self.steps):
            perturb_img.requires_grad = True
            trans_pertueb_images = self.trans(perturb_img)
            outputs = self.model(trans_pertueb_images)
            # if outputs.argmax(1).eq(target):
            #     perturb_img=perturb_img.clone().detach()
            #     break
            if self.target:
            # far from the deep representation
                cost = -self.criterion(outputs, target)
            else:
                cost=self.criterion(outputs,target)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, perturb_img, retain_graph=False, create_graph=False
            )[0]

            perturb_img = perturb_img.detach() + self.alpha * grad.sign()
            delta = torch.clamp(perturb_img - images, min=-self.eps, max=self.eps)
            perturb_img = torch.clamp(images + delta, min=0, max=1).detach()

        return perturb_img


#!/usr/bin/python
# author eson
