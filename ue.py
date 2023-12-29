#!/usr/bin/python
# author eson
from argparse import ArgumentParser

import numpy as np

import os

from attacks import *
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

np.random.seed(seed=3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

config = load_conf(args.conf_path)
config.attack.eps /= 255
config.attack.alpha /= 255

config.drm_attack.eps /= 255
config.drm_attack.alpha /= 255

# path where result was saved
res_folder = config.result_dir
image_save_dir = os.path.join(res_folder, "images")
make_dir(res_folder)
make_dir(image_save_dir)


def ue():
    criterion = instantiate(config.criterion)
    hl_criterion = instantiate(config.hl_criterion)
    model = instantiate(config.model).to(device)
    if config.data_parallel:
        model = torch.nn.DataParallel(model)

    optimizer = instantiate(config.optimizer, model.parameters())

    # 有目标投毒攻击PGD，目标为原本的正确标签
    attack = instantiate(config.attack, model)
    attack.set_mode_targeted_by_function(target_map_function=lambda images, labels: labels)

    drm_attack = instantiate(config.drm_attack, model=model, criterion=criterion)

    dataset = get_dataset(
        "ue_gen",
        config.db_name,
        noise_rate=config.noise_rate,
        train_aug=False,
        noise_idx=torch.load(config.noise_path)
    )

    train_dataset = dataset["train"]["ll"] if type(dataset["train"]) == dict else dataset["train"]
    # use subset to train for steps to update \theta
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=True,
        num_workers=20
    )

    # use to generate noise
    order_train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=20
    )

    if config.noise_rate:
        hl_loader = DataLoader(
            dataset=dataset["train"]["hl"],
            batch_size=512,
            shuffle=False,
            drop_last=False,
            num_workers=20
        )
    else:
        hl_loader = None
    logger = Logger(0, log_loss=False)
    ng = NoiseGenerator(train_loader, order_train_loader, hl_loader, model, 10, attack, drm_attack)
    # Save noise
    perturbation = ng.ue(optimizer, criterion, 0.05, logger)
    print(f"Perturbation saved at {os.path.join(res_folder, 'perturbation.pt')}. ")
    torch.save(perturbation, os.path.join(res_folder, "perturbation.pt"))

    hl_perturbation = ng.deep_representation_manipulation(hl_criterion)
    print(f"High loss samples Perturbation saved at {os.path.join(res_folder, 'hl_perturbation.pt')}. ")
    torch.save(hl_perturbation, os.path.join(res_folder, "hl_perturbation.pt"))


if __name__ == '__main__':
    ue()
