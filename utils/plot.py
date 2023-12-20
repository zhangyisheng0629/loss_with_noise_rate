#!/usr/bin/python
# author eson

import matplotlib.pyplot as plt

import numpy as np

from train.logger import Logger


def pl_plot(save_path, logger, eval_acc=0):
    for i in [
        "clean_loss",
        "noise_loss",
        "clean_num",
        "noise_num",
    ]:
        assert hasattr(logger, i)

    x = np.linspace(0, 1., 11)
    y = [list() for _ in range(len(x))]

    start, end = 0, 1 + 1e-3
    step = 0.1
    for c_l, n_l, c_n, n_n in zip(logger.clean_loss,
                                  logger.noise_loss,
                                  logger.clean_num,
                                  logger.noise_num):

        noise_rate = n_n / (c_n + n_n)
        for i in range(1, len(x)):
            if x[i - 1] <= noise_rate < x[i]:
                y[i - 1].append((c_l + n_l) / (c_n + n_n))
    avg_y = [0 for _ in range(len(x))]
    for i, l in enumerate(y):
        if l:
            avg_y[i] = sum(l) / len(l)
        else:
            avg_y[i] = 0
    fig, ax = plt.subplots()
    ax.set_xlabel("Noise ratio")
    ax.set_ylabel("Average cross entropy loss")
    ax.set_title(f"Epoch {logger.cur_epoch}, val acc {eval_acc:.2f}")
    plt.xticks(x)
    plt.bar(x, avg_y, width=0.1, align="edge", color="r")
    for a, b in zip(x, avg_y):
        if b:
            plt.text(a, b, f"{b:.2f}")

    fig.savefig(save_path)
    # fig.clo
    pass


def pl_plot2(path, logger: Logger):
    pass


if __name__ == '__main__':
    logger = Logger(cur_epoch=0, train=True)
    logger.clean_loss = [i for i in range(1, 11)]
    logger.noise_loss = [i for i in range(9, -1, -1)]
    logger.clean_num = [i for i in range(1, 11)]
    logger.noise_num = [i for i in range(9, -1, -1)]
    pl_plot("", logger)
