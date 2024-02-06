#!/usr/bin/python
# author eson
# !/usr/bin/python
# author eson
import os
from argparse import ArgumentParser

import numpy as np
import torch
from mlconfig import instantiate

from train_utils.evaluator import Evaluator
from train_utils.train_model import train_clean_model
from utils import *

from attacks import *
from models import *
from train_utils.logger import Logger
from train_utils.trainer import Trainer
from utils.common_utils import load_model, make_dir
from utils.config import load_conf
from utils.get import get_dataset, get_dataset_, get_transform

from torch.utils.data import DataLoader
from utils.common_utils import ConfusionMatrixDrawer
from utils.noise_generator import NoiseGenerator, UENoiseGenerator

arg_parser = ArgumentParser()
arg_parser.add_argument("--conf_path", type=str, default="", help="Config file path.")
args = arg_parser.parse_args()
config = load_conf(args.conf_path)
config.attack.eps /= 255
config.attack.alpha /= 255
config.hl_attack.eps /= 255
config.hl_attack.alpha /= 255

for k, v in dict(config).items():
    print(f"{k} : {v}")
SEED = 1
np.random.seed(seed=SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
res_folder = config.log_dir
make_dir(res_folder)


def init_(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def perturb():
    outer_criterion = instantiate(config.outer_criterion)

    model = instantiate(config.model).to(device)
    model.apply(init_)

    if config.data_parallel:
        model = torch.nn.DataParallel(model)

    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.scheduler, optimizer)

    if config.attack.name == "UEPGD":
        attack = instantiate(config.attack, model=model)
        attack.set_mode_targeted_by_function(target_map_function=lambda images, labels: labels)
    elif config.attack.name == "EOTUEPGD":
        trans = get_transform(config.db_name, is_tensor=True)["train_transform"]
        attack = instantiate(config.attack, model=model, trans=trans)
        attack.set_mode_targeted_by_function(target_map_function=lambda images, labels: labels)
    # dataset = get_dataset(
    #     config.db_name,
    #     train_aug=False
    # )
    dataset = get_dataset_(
        "ue_gen",
        config.db_name,
        noise_rate=0.2,
        train_aug=False,
        noise_path=config.noise_path,
    )
    train_dataset = dataset["train"]["ll"] if type(dataset["train"]) == dict else dataset["train"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=True,
        num_workers=8,
    )

    order_train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )
    val_loader = DataLoader(
        dataset["val"],
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    if config.ckpt_dir:
        ckpt_dir = config.ckpt_dir
        ckpt = load_model(ckpt_dir, model, optimizer, scheduler, )

        start_epoch = ckpt["epoch"] + 1
        print(f"File {ckpt_dir} loaded! Last training process was epoch {ckpt['epoch']}. ")
    else:
        start_epoch = 0
        print("Train from scratch. ")

    logger = Logger(0, log_loss=False)

    ng = UENoiseGenerator(train_loader, order_train_loader, model, config.train_steps, attack)
    perturbation = ng.ue(optimizer, outer_criterion, config.stop_error, logger,
                         True, val_loader, config.gen_on, num_classes=config.model["num_classes"],
                         trans=trans,res_folder=res_folder)

    # Save noise
    print(f"Perturbation saved at {os.path.join(res_folder, 'perturbation.pt')}. ")
    torch.save(perturbation, os.path.join(res_folder, "perturbation.pt"))

    if config.hl:
        hl_loader = DataLoader(dataset["train"]["hl"], batch_size=512, shuffle=False, drop_last=False, num_workers=8)

        # model optim criterion
        clean_model = instantiate(config.clean_model).to(device)
        ll_optimizer = instantiate(config.ll_optimizer, clean_model.parameters())
        clean_model_criterion = instantiate(config.clean_model_criterion)

        # TODO:return
        clean_model = train_clean_model(dataset["train"]["ll"], hl_loader, 0.8, clean_model, ll_optimizer,
                                        clean_model_criterion,
                                        5)
        clean_model.eval()
        hl_criterion = instantiate(config.hl_criterion)

        poison_list = {i: i + 1 for i in range(0, 10, 2)}
        poison_list.update({i: i - 1 for i in range(1, 10, 2)})

        hl_attack = instantiate(config.hl_attack, model=model,
                                criterion=hl_criterion,
                                target_map=lambda images, labels: clean_model(images),
                                # target_map=lambda images, labels: torch.Tensor(
                                #     [poison_list[l.item()] for l in model(images).argmax(1)]).type(
                                #     torch.int64),
                                target=True,
                                )

        hl_perturbation = ng.deep_representation_manipulation(hl_loader, hl_attack)
        print(f"High loss samples Perturbation saved at {os.path.join(res_folder, 'hl_perturbation.pt')}. ")
        torch.save(hl_perturbation, os.path.join(res_folder, "hl_perturbation.pt"))


if __name__ == '__main__':
    # clean_model = instantiate(config.model)
    # images = torch.rand(size=(5, 3, 32, 32))
    # labels = clean_model(images).argmax(1)
    # print(labels.dtype)
    # poison_list = {i: i + 1 for i in range(0, 10, 2)}
    # poison_list.update({i: i - 1 for i in range(1, 10, 2)})
    # target_map = lambda images, labels: torch.Tensor([poison_list[l.item()] for l in labels]).type(torch.int64)
    #
    # print(target_map(images, labels))

    perturb()
