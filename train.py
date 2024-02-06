import os
from argparse import ArgumentParser

import numpy as np
import torch
from mlconfig import instantiate

from train_utils.evaluator import Evaluator
from train_utils.trainer import Trainer
from utils import *

from attacks import *
from models import *
from train_utils.logger import Logger

from utils.common_utils import load_model, ConfusionMatrixDrawer, save_model, make_dir
from utils.config import load_conf
from utils.get import get_dataset, get_dataset_

from torch.utils.data import DataLoader

from utils.noise_generator import NoiseGenerator

arg_parser = ArgumentParser()
arg_parser.add_argument("--conf_path", type=str, default="", help="Config file path.")
args = arg_parser.parse_args()
config = load_conf(args.conf_path)
for k, v in dict(config).items():
    print(f"{k} : {v}")
SEED=1
np.random.seed(seed=SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
res_folder = config.log_dir
make_dir(res_folder)
def init_(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

def train():
    # model,criterion,optimizer,scheduler
    criterion = instantiate(config.criterion)
    model = instantiate(config.model).to(device)
    model.apply(init_)
    if config.data_parallel:
        model = torch.nn.DataParallel(model)
    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.scheduler, optimizer)

    # dataset,dataloader
    # dataset = get_dataset(
    #     config.db_name,
    #     train_aug=True,
    #     perturbfile_path=config.perturbfile_path
    # )
    dataset = get_dataset_(
        stage="ue_train",
        db_name=config.db_name,
        noise_rate=config.noise_rate,
        train_aug=True,
        noise_path=config.noise_path,
        perturbfile_path=config.perturbfile_path,
        hl_perturbfile_path=config.hl_perturbfile_path,
        poison_rate=config.poison_rate
    )
    train_loader = DataLoader(
        dataset["train"],
        128,
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

    train_acc_logger = Logger(start_epoch=start_epoch, log_loss=False)
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
    # train_acc_logger在一个 epoch 之后计算 train acc
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

    cmd = ConfusionMatrixDrawer(config.model.num_classes)
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

    for epoch in range(start_epoch, config.total_epoch):
        print("Train")
        trainer.train(train_on="clean")
        # print("Train eval")
        # eval_res = train_evaluator.eval()
        print("Val eval")
        evaluator.eval(cmd,eval_on="clean")

        # reset
        train_evaluator.logger.new_epoch()
        trainer.logger.new_epoch()
        evaluator.logger.new_epoch()

        scheduler.step()
        save_model(res_folder, epoch, model, optimizer, scheduler)
        print(f'Ckpt saved at {os.path.join(res_folder, "state_dict.pth")}')
        # confusion matrix
        if cmd:
            if (epoch + 1) % 5 == 0:
                print(cmd.confusion)
                cmd.draw(title=f"UE train, epoch{epoch}. ")
            cmd.reset()


if __name__ == '__main__':
    train()
