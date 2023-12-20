#!/usr/bin/python
# author eson
import os
import platform


class DataDir(object):
    def __init__(self):
        op_sys = platform.system()
        if op_sys == "Windows":
            PRETRAINED_MODELS_DIR = "F:/pretrained_models"
            DATASETS_DIR = "F:/datasets"
        elif op_sys == "Linux":
            PRETRAINED_MODELS_DIR = "/users/uestc1/zys/pretrained_models"
            DATASETS_DIR = "/users/uestc1/zys/Datasets"
        self.base_dir = DATASETS_DIR

    def get_dir(self, db_name):
        if db_name in ["tiny_imagenet", "imagenet_mini"]:
            db_name = "imagenet"

        return os.path.join(self.base_dir, db_name)
