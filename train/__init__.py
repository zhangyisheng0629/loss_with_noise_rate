#!/usr/bin/python
# author eson
import mlconfig

from . import loss_logger

mlconfig.register(loss_logger.LossLogger)
mlconfig.register(loss_logger.TopkLossLogger)
mlconfig.register(loss_logger.ThreLossLogger)