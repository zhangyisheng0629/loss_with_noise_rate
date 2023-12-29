#!/usr/bin/python
# author eson
import torch


class uniform_generator():
    def __init__(self,model):

        self.attack=attack
        self.criterion=criterion
        self.model=model
        self.steps=steps
        self.eps=eps
        self.alpha=alpha
        pass


    def generate(self):
        pass


    def add_noise(self,images):


        logits=self.model(images)

        perturb_img=images.clone().detach()
        for _ in self.steps:
            perturb_img.requires_grad = True
            outputs = self.model(perturb_img)


            cost = -self.criterion(outputs, logits)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, perturb_img, retain_graph=False, create_graph=False
            )[0]

            perturb_img = perturb_img.detach() + self.alpha * grad.sign()
            delta = torch.clamp(perturb_img - images, min=-self.eps, max=self.eps)
            perturb_img = torch.clamp(images + delta, min=0, max=1).detach()

        return perturb_img
