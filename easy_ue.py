#!/usr/bin/python
# author eson
import os
from argparse import ArgumentParser

import numpy as np
import torch
from mlconfig import instantiate
from utils import *
from train import *
from attacks import *
from models import *
from train.logger import Logger
from train.trainer import Trainer
from utils.common_utils import load_model
from utils.config import load_conf
from utils.get import get_dataset

from torch.utils.data import DataLoader

from utils.noise_generator import NoiseGenerator

arg_parser = ArgumentParser()
arg_parser.add_argument("--conf_path", type=str, default="", help="Config file path.")
args = arg_parser.parse_args()

config = load_conf(args.conf_path)
np.random.seed(seed=3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

res_folder = config.log_dir

def perturb():
    criterion = instantiate(config.criterion)
    model = instantiate(config.model).to(device)
    if config.data_parallel:
        model = torch.nn.DataParallel(model)

    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.scheduler, optimizer)
    attack = instantiate(config.attack, model)
    attack.set_mode_targeted_by_function(target_map_function=lambda images, labels: labels)

    dataset = get_dataset(
        "select",
        config.db_name,
        noise_rate=0,
        train_aug=False,
    )
    train_loader = DataLoader(
        dataset["train"],
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    order_train_loader = DataLoader(
        dataset["train"],
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    pass
    val_loader = DataLoader(
        dataset=dataset["val"],
        batch_size=512,
        shuffle=False,
        num_workers=40
    )
    if "ckpt_dir" in config:
        ckpt_dir = config.ckpt_dir
        ckpt = load_model(ckpt_dir, model, optimizer, scheduler, )

        start_epoch = ckpt["epoch"] + 1
        print(f"File {ckpt_dir} loaded! Last training process was epoch {ckpt['epoch']}. ")
    else:
        start_epoch = 0
        print("Train from scratch. ")

    logger = Logger(0, log_loss=False)
    ng = NoiseGenerator(train_loader, order_train_loader,None, model, 10, attack)
    perturbation = ng.ue(optimizer, criterion, 0.01, logger)
    # Ckpt
    # Save noise
    print(f"Perturbation saved at {os.path.join(res_folder, 'perturbation.pt')}. ")
    torch.save(perturbation, os.path.join(res_folder, "perturbation.pt"))
    # trainer=Trainer
    # train()
    # eval()
    # recog_
def train():

    pass
if __name__ == '__main__':
    if os.path.exists(os.path.join(res_folder, "perturbation.pt")):
        perturb()
    train()