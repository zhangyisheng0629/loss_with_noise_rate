#!/usr/bin/python
# author eson
import mlconfig

from . import criterion

mlconfig.register(criterion.MixCrossEntropyLoss)
mlconfig.register(criterion.SCELoss)
mlconfig.register(criterion.SCANLoss)
