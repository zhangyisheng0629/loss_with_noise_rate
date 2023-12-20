#!/usr/bin/python
# author eson
import numpy as np
import torch


class LossLogger():
    def __init__(self, n, poison_idx, ):
        self.n = n
        self.poison_idx = poison_idx
        self.pos = 0
        self.loss = torch.FloatTensor(n)
        self.recog_idx=None
        pass
    def update(self, loss: torch.FloatTensor):
        b = loss.shape[0]
        self.loss[self.pos:self.pos + b].copy_(loss.detach())
        self.pos += b
    def reset(self):
        self.pos = 0

    def recog_noise(self):
        raise NotImplementedError("You need to implement the recog_noise function. ")


class ThreLossLogger(LossLogger):
    def __init__(self, n, poison_idx, thre_loss=1):
        super().__init__(n, poison_idx)
        self.thre_loss = torch.tensor(thre_loss)



    def recog_noise(self, ):
        """
        将loss超过thre的样本假设为噪声
        Args:
            k:
        Returns:

        """
        poison_idx = self.poison_idx

        recog_idx = torch.nonzero(torch.gt(self.loss, self.thre_loss))
        recog_idx_np = recog_idx.numpy()
        tp = sum([i in poison_idx for i in recog_idx_np])
        # precision,recall
        print(f"recog len {len(recog_idx)} ")
        self.recog_idx=recog_idx
        return tp / len(recog_idx), tp / len(poison_idx)


class TopkLossLogger(LossLogger):
    def __init__(self, n, poison_idx, topk_rate=0.2, ):
        super().__init__(n, poison_idx)
        self.k = int(topk_rate * n)

    def topk_loss(self, k: int):
        topk, topk_idx = torch.topk(self.loss, k=k, largest=True)
        return topk, topk_idx

    def recog_noise(self, ):
        """
        将loss最大的k的样本假设为噪声
        Args:
            k:
        Returns:

        """
        k = self.k
        poison_idx = self.poison_idx
        topk, topk_idx = self.topk_loss(k)
        topk_idx_numpy = topk_idx.numpy()
        tp = sum([i in poison_idx for i in topk_idx_numpy])
        # precision,recall
        self.recog_idx=topk_idx
        return tp / k, tp / len(poison_idx)


if __name__ == '__main__':
    ll = TopkLossLogger(10, poison_idx=np.array([1, 2, 3, 4, 5, 8, 9, 10]))
    for i in range(5):
        ll.update(torch.Tensor([i, i]))
    print(ll.loss)
    print(ll.topk_loss(2))
    print(ll.recog_noise( ))
