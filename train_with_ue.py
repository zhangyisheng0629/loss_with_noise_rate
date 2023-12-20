#!/usr/bin/python
# author eson
from argparse import ArgumentParser

import numpy as np

import os

from models import *
from train.evaluator import Evaluator
from train.trainer import Trainer
from utils import *
from train import *
from mlconfig import instantiate
from torch.utils.data import DataLoader

from train.logger import Logger
from utils.common_utils import make_dir, load_model, save_model
from utils.config import load_conf
from utils.get import get_dataset, get_subset
from utils.noise_generator import NoiseGenerator

arg_parser = ArgumentParser()
arg_parser.add_argument("--conf_path", type=str, default="", help="Config file path.")
args = arg_parser.parse_args()

config = load_conf(args.conf_path)
np.random.seed(seed=0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

perturbfile_dir=config.perturbfile_dir
# path where result was saved
res_folder = config.log_dir
image_save_dir = os.path.join(res_folder, "images")
make_dir(res_folder)
make_dir(image_save_dir)


def train():
    criterion = instantiate(config.criterion)
    model = instantiate(config.model).to(device)
    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.scheduler, optimizer)
    dataset = get_dataset(
        "ue_train",
        "cifar-10",
        train_aug=True,
        noise_rate=config.noise_rate,
        noise_idx=torch.load(os.path.join(config.noise_path, "noise_idx.pt")),
        perturbfile_path=os.path.join(res_folder, "perturbation.pt")
    )

    # TODO:add the poison dataset in get_dataset() function

    train_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    train_inorder_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=40
    )
    val_loader = DataLoader(
        dataset=dataset["val"],
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    if "ckpt_dir" in config:
        ckpt_dir = config.ckpt_dir
        ckpt = load_model(ckpt_dir, model, optimizer, scheduler, )

        start_epoch = ckpt["epoch"] + 1
        print(f"File {ckpt_dir} loaded! Last training process was epoch {ckpt['epoch']}. ")
    else:
        start_epoch = 0
        print("Train from scratch. ")
    # res_folder:|/result/cifar-10/time()
    #            |  images/
    #            |  log.json
    #            |  model.ckpt

    train_logger = Logger(start_epoch=start_epoch, log_loss=False)
    val_logger = Logger(start_epoch=start_epoch, log_loss=False)

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
        criterion,
        loss_logger=None,
        args=args,
        cur_epoch=start_epoch,
        total_epoch=config.total_epoch,
        device=device
    )
    # train_acc_logger在一个 epoch 之后计算 train acc
    train_acc_logger = Logger(start_epoch=start_epoch, log_loss=False)
    train_evaluator = Evaluator(
        train_acc_logger,
        train_inorder_loader,
        model,
        criterion=criterion,
        loss_logger=None,
        args=args,
        cur_epoch=start_epoch,
        total_epoch=config.total_epoch,
        device=device
    )

    for epoch in range(start_epoch, config.total_epoch):
        print("Train")
        trainer.train(train_on="clean")
        print("Train eval")
        eval_res = train_evaluator.eval()
        print("Val eval")
        evaluator.eval()

        # save log file
        acc_dict = {"train_acc": train_acc_logger.cal_acc(),
                    "val_acc": evaluator.logger.cal_acc()}
        # trainer.logger.save(os.path.join(res_folder, "train_log.json"),
        #                     acc_dict)
        # save image
        # plot_save_path = os.path.join(image_save_dir, f"epoch{epoch}.png")
        # pl_plot(plot_save_path, trainer.logger, acc_dict["val_acc"])

        # reset
        train_evaluator.logger.new_epoch()

        # train_evaluator.loss_logger.reset()
        trainer.logger.new_epoch()
        evaluator.logger.new_epoch()

        # TODO:add function to the utils.plot file

        scheduler.step()
        save_model(res_folder, epoch, model, optimizer, scheduler)
        print(f'Ckpt saved at {os.path.join(res_folder, "state_dict.pth")}')







if __name__ == '__main__':
    train()