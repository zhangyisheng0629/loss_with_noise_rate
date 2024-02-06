#!/usr/bin/python
# author eson
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from torch.utils.data import DataLoader

# from end_plot import json2excel
from utils.get import get_dataset, get_dataset_
# from utils.plot import pl_plot
from models import *
from utils import *
from train_utils import *
from argparse import ArgumentParser
from train_utils.trainer import Trainer
from train_utils.logger import Logger
from train_utils.evaluator import Evaluator
from utils.config import load_conf
from mlconfig import instantiate
from utils.common_utils import make_dir, save_model, load_model, save_idx

arg_parser = ArgumentParser()
arg_parser.add_argument("--conf_path", type=str, default="", help="Config file path.")
args = arg_parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SEED = 1
# random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

config = load_conf(args.conf_path)
for k, v in dict(config).items():
    print(f"{k} : {v}")


def main():
    criterion = instantiate(config.criterion)
    model = instantiate(config.model).to(device)
    if config.data_parallel:
        model = torch.nn.DataParallel(model)
    # 日期 时间
    res_folder = config.result_dir
    image_save_dir = os.path.join(res_folder, "images")
    make_dir(res_folder)
    make_dir(image_save_dir)

    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.scheduler, optimizer)
    dataset = get_dataset_(
        "select",
        config.db_name,
        noise_rate=config.noise_rate,
        train_aug=True,
    )
    train_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=32
    )
    train_inorder_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=32
    )
    val_loader = DataLoader(
        dataset=dataset["val"],
        batch_size=128,
        shuffle=False,
        num_workers=32
    )
    # TODO: check if exists ckpt, load the trained process from saved ckpt

    if config.ckpt_dir:
        ckpt_dir = config.ckpt_dir
        ckpt = load_model(ckpt_dir, model, optimizer, scheduler, )

        start_epoch = ckpt["epoch"] + 1
        print(f"File {ckpt_dir} loaded! Last training process was epoch {ckpt['epoch']}. ")
    else:
        start_epoch = 0
        print("Train from scratch. ")

    train_logger = Logger(start_epoch=start_epoch, log_loss=False)
    val_logger = Logger(start_epoch=start_epoch, log_loss=False)
    # train_acc_logger在一个 epoch 之后计算 train acc
    train_acc_logger = Logger(start_epoch=start_epoch, log_loss=True)
    if config.noise_rate:
        noise_recognizer = instantiate(
            config.loss_logger,
            n=len(dataset["train"]),
            poison_idx=dataset["train"].poison_idx,
        )
    else:
        noise_recognizer = None
    # noise_recognizer = LossLogger(
    #     n=len(dataset["train"]),
    #     poison_idx=dataset["train"].poison_idx,
    #     topk_rate=config.topk_rate
    # )
    trainer = Trainer(
        train_logger,
        train_loader,
        model,
        optimizer,
        criterion,
        args,
        start_epoch,
        config.total_epoch,
        device
    )

    evaluator = Evaluator(
        val_logger,
        val_loader,
        model,
        None,
        None,
        args,
        start_epoch,
        config.total_epoch,
        device
    )

    train_evaluator = Evaluator(
        train_acc_logger,
        train_inorder_loader,
        model,
        criterion=criterion,
        loss_logger=noise_recognizer,
        args=args,
        cur_epoch=start_epoch,
        total_epoch=config.total_epoch,
        device=device
    )

    for epoch in range(start_epoch, config.total_epoch):
        print("Train")
        trainer.train(train_on="noise")  # train on noise
        print("Train eval")
        eval_res = train_evaluator.eval(eval_on="noise", log_loss=True)  # train set  eval on noise
        print("Val eval")
        evaluator.eval(eval_on="clean", log_loss=False)  # val set   eval on clean

        # save log file
        acc_dict = {"train_acc": train_acc_logger.cal_acc(),
                    "val_acc": evaluator.logger.cal_acc()}
        train_evaluator.logger.save(os.path.join(res_folder, "train_log.json"),
                                    acc_dict,
                                    **eval_res
                                    )
        # save image
        plot_save_path = os.path.join(image_save_dir, f"epoch{epoch}.png")
        # pl_plot(plot_save_path, train_evaluator.logger, acc_dict["val_acc"])

        # reset
        train_evaluator.logger.new_epoch()
        trainer.logger.new_epoch()
        evaluator.logger.new_epoch()
        train_evaluator.loss_logger.reset()
        # TODO:add function to the utils.plot file

        scheduler.step()
        save_model(res_folder, epoch, model, optimizer, scheduler)
        save_idx(res_folder, train_evaluator.loss_logger.recog_idx)
        print(f'Ckpt saved at {os.path.join(res_folder, "state_dict.pth ")}\n'
              f'High loss sample idx saved at {os.path.join(res_folder, "noise_idx.pt")}')
        # Plot loss accuracy fig per 40 epochs
        # if "plot_freq" in config and not (epoch + 1) % config.plot_freq:
        #     json2excel(os.path.join(res_folder, "train_log.json"), '')


if __name__ == '__main__':
    main()
