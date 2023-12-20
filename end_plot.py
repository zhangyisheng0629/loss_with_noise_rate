#!/usr/bin/python
# author eson
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def json2excel(log_path, save_path):
    train_acc = []
    val_acc = []

    precision=[]
    recall=[]

    clean_avg_loss = []
    noise_avg_loss = []
    with open(log_path, "r+") as f:

        while True:
            row = f.readline()
            if row:
                # for k,v in row.ite
                row = json.loads(row)
                train_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))

                precision.append(row["precision"])
                recall.append(row["recall"])

                clean_avg_loss.append(sum(row["clean_loss"]) / sum(row["clean_num"]))
                noise_avg_loss.append(sum(row["noise_loss"]) / sum(row["noise_num"]))
            else:
                break
    x = list(range(len(train_acc)))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch /200 ")
    ax1.set_ylabel("average loss.")
    # plt.yticks(np.linspace(0.3,0.7,8))
    ax2 = ax1.twinx()
    # ax2.set_yticks(np.linspace(0.0,1.0,2))
    ax2.set_ylabel("Acc /%")

    ax2.plot(x, train_acc, color="orange", label="train_acc(noise label)")
    ax2.plot(x, val_acc, color="b",label="val_acc")

    ax2.plot(x, precision, label="precision")
    ax2.plot(x, recall,color="g", label="recall")

    ax1.plot(x, clean_avg_loss,linestyle="--", label="clean_avg_loss")
    ax1.plot(x, noise_avg_loss,linestyle="--", label="noise_avg_loss")


    ax1.legend(loc="center right")
    ax2.legend(loc="lower left")

    plt.show()

    # df=pd.DataFrame.from_dict()
    # df.to_excel("./results/cifar-10/train_log.xlsx")
    #     log_dict=json.loads(f)


if __name__ == '__main__':
    json2excel("./results/cifar-10/train_log.json", "")
    json2excel("./results/svhn/train_log.json", "")
    # json2excel("./results/tiny_imagenet/train_log.json", "")
    json2excel("./results/imagenet_mini/train_log.json", "")
