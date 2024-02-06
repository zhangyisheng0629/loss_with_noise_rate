import torch
from torch import nn

POISON_TARGET_FUN = [lambda images, labels: (labels + 1) % 10]


class MixCEPGD(object):
    def __init__(self, model: torch.nn.Module, eps, alpha, steps,
                 random_start=True, poison_target_fun_num=0, w1=1.0, w2=0.0):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.poison_target_fun = POISON_TARGET_FUN[poison_target_fun_num]  # lambda images, labels: (labels + 1) % 10
        self.w1 = w1 / (w1 + w2)
        self.w2 = w2 / (w1 + w2)
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None
            print("Failed to set device automatically, please try set_device() manual.")

    def __call__(self, images, labels, noise):
        r"""
        Overridden.
        """
        loss = nn.CrossEntropyLoss()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        target_labels = self.poison_target_fun(images, labels)

        images = images + noise
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss

            cost = -(self.w1 * loss(outputs, labels) + self.w2 * loss(outputs, target_labels))

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

