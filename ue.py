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

np.random.seed(seed=1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = load_conf(args.conf_path)
config.attack.eps /= 255
config.attack.alpha /= 255

# path where result was saved
res_folder = config.log_dir
image_save_dir = os.path.join(res_folder, "images")
make_dir(res_folder)
make_dir(image_save_dir)


def ue():
    criterion = instantiate(config.criterion)
    model = instantiate(config.model).to(device)
    optimizer = instantiate(config.optimizer, model.parameters())

    # 有目标投毒攻击PGD，目标为原本的正确标签
    attack = instantiate(config.attack, model)
    attack.set_mode_targeted_by_function(target_map_function=lambda images, labels: labels)

    dataset = get_dataset(
        "ue_gen",
        "cifar-10",
        noise_rate=0,
        train_aug=False,

        # noise_idx=torch.load(os.path.join(config.noise_path, "noise_idx.pt"))
    )

    # subset_idx = torch.arange(50000)[torch.where(torch.isin(torch.arange(50000), idx) == False, True, False)]
    # train_subset = get_subset(dataset["train"], subset_idx)
    # use subset to train for steps to update \theta
    train_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=512,
        shuffle=False,
        drop_last=True,
        num_workers=0
    )

    # use to generate noise
    order_train_loader = DataLoader(
        dataset=dataset["train"],
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    logger = Logger(0, log_loss=False)
    ng = NoiseGenerator(train_loader, order_train_loader, model, 10, attack)
    perturbation = ng.ue(optimizer, criterion, 0.01, logger)

    # Ckpt

    # NoiseGenerator is a tool to generate noise for arbitratry

    # Save noise
    print(f"Perturbation saved at {os.path.join(res_folder, 'perturbation.pt')}. ")
    torch.save(perturbation, os.path.join(res_folder, "perturbation.pt"))


if __name__ == '__main__':
    ue()
