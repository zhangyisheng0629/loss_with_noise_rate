#!/usr/bin/python
# author eson

import json


class Logger:
    def __init__(self, start_epoch, type_dict=None, log_loss=True, mode=None):
        """

        Args:
            start_epoch:
            type_dict:
            log_loss:
            mode: log_mode,"a" means record accuracy
                           "l" means record loss
        """
        # base：记录预测正确的样本，计算准确率，log_loss：记录损失，计算平均损失
        assert mode in ["a", "l", "al", None]
        self.mode = mode
        self.log_loss = log_loss
        if self.log_loss:
            self.clean_loss = []
            self.noise_loss = []
            self.clean_num = []
            self.noise_num = []

            self.total_clean_num = 0
            self.total_noise_num = 0
        self.cur_epoch = start_epoch

        self.total_num = 0
        self.correct_num = []

    def log(self, payload):
        if self.log_loss:
            self.clean_loss.append(payload["clean_loss"])
            self.noise_loss.append(payload["noise_loss"])
            self.clean_num.append(payload["clean_num"])
            self.noise_num.append(payload["noise_num"])
            self.total_clean_num += payload["clean_num"]
            self.total_noise_num += payload["noise_num"]

        self.correct_num.append(payload["correct_num"])
        self.total_num += payload["batch_num"]

    def cal_avg_loss(self, key):
        if key == "clean":
            return sum(self.clean_loss) / self.total_clean_num if self.total_clean_num else 0
        elif key == "noise":
            return sum(self.noise_loss) / self.total_noise_num if self.total_noise_num else 0
        else:
            raise KeyError(f"No {key} key to calculate loss. ")

    def reset(self):
        if self.log_loss:
            self.clean_loss.clear()
            self.noise_loss.clear()
            self.clean_num.clear()
            self.noise_num.clear()
            self.total_clean_num = 0
            self.total_noise_num = 0
        self.total_num = 0
        self.correct_num.clear()

    def new_epoch(self):
        self.cur_epoch += 1

        self.reset()

    def display_str(self):
        return

    def save(self, file_path, acc_dict, **kwargs):
        json_res = {
            "epoch": self.cur_epoch,
            "train_acc": round(acc_dict["train_acc"], 2),
            "val_acc": round(acc_dict["val_acc"], 2), }
        if self.log_loss:
            json_res.update({
                "clean_loss": self.clean_loss,
                "noise_loss": self.noise_loss,
                "clean_num": self.clean_num,
                "noise_num": self.noise_num
            })
        json_res.update(kwargs)
        with open(file_path, "a+") as f:
            f.write(json.dumps(json_res) + "\n")

    def cal_acc(self):
        acc = sum(self.correct_num) / self.total_num
        return acc
