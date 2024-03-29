#!/usr/bin/python
# author eson
import os

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class ConfusionMatrixDrawer():
    def __init__(self, class_number):
        self.confusion = torch.zeros([class_number, class_number])
        self.class_number = class_number

    def set_class_number(self, n):
        self.class_number = n

    def update(self, label: torch.Tensor, pred: torch.Tensor):
        for i, j in zip(label, pred):
            self.confusion[i][j] += 1

    def reset(self):
        self.confusion = torch.zeros([self.class_number, self.class_number])

    def draw(self,title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.confusion.numpy())
        fig.colorbar(cax)
        all_categories = list("abcdefghij")
        # Set up axes
        ax.set_xticklabels([''] + all_categories, rotation=90)
        ax.set_yticklabels([''] + all_categories)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_title(title)
        # sphinx_gallery_thumbnail_number = 2
        plt.show()

        pass

    def update_draw(self, loader,model,device,title=None):

        for i,batch in enumerate(loader):
            X,y=batch
            X,y=X.to(device),y.to(device)
            pred=model(X).argmax(1)

            self.update(y,pred)

        self.draw(title)


def make_dir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)


device = "cuda" if torch.cuda.is_available() else "cpu"


def save_idx(dir, idx):
    save_path = os.path.join(dir, 'noise_idx.pt')
    torch.save(idx, save_path)

    return


def save_model(dir, epoch, model, optimizer, scheduler=None, **kwargs):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    for key, value in kwargs.items():
        state[key] = value
    save_path = os.path.join(dir, 'state_dict.pth')
    torch.save(state, save_path)
    return


def load_model(dir, model, optimizer=None, scheduler=None, **kwargs):
    # Load Torch State Dict

    filename = os.path.join(dir, 'state_dict.pth')
    checkpoints = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'])
    if optimizer is not None and checkpoints['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    if scheduler is not None and checkpoints['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    return checkpoints


def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise ('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[x1: x2, y1: y2, :] = noise
    return mask


if __name__ == '__main__':
    cmd = ConfusionMatrixDrawer(10)
    for i in tqdm(range(391)):
        label = torch.randint(0, 10, [128, ])
        pred = torch.randint(0, 10, [128, ])
        cmd.update(label, pred)
    print(cmd.confusion)

    cmd.draw()
