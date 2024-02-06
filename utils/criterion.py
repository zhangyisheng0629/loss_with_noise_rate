#!/usr/bin/python
# author eson
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(reduction):
    c = nn.CrossEntropyLoss(reduction=reduction)
    return c


class MixCrossEntropyLoss(nn.Module):
    def __init__(self, w1, w2, target_labels_fun=lambda labels: (labels + 1) % 10):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.target_labels_fun = target_labels_fun
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, labels):
        target_label = self.target_labels_fun(labels)
        loss1 = self.ce(output, labels)
        loss2 = self.ce(output, target_label)
        loss = self.w1 * loss1 + self.w2 * loss2
        return loss.mean()
class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        """

        Args:
            alpha,beta: for cifar10-0.1,1.0,cifar100-6.0,1.0

            num_classes:
        """
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-8)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0,ce_weight=0.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.ce = nn.CrossEntropyLoss()
        self.entropy_weight = entropy_weight  # Default = 2.0
        self.ce_weight = ce_weight
    def forward(self, output, labels):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = output.size()
        output_prob = self.softmax(output)

        # Similarity in output space


        ce_loss = self.ce(output, labels)

        # Entropy loss
        entropy_loss = entropy(torch.mean(output_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = self.ce_weight*ce_loss - self.entropy_weight * entropy_loss

        return total_loss
if __name__ == '__main__':
    kwargs = {"reduction": "none"}
    c = get_criterion(**kwargs)
    pass
