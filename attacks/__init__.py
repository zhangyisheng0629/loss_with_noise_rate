#!/usr/bin/python
# author eson
from torchattacks import PGD
import mlconfig
from . import drm,ue_pgd
mlconfig.register(PGD)
mlconfig.register(drm.DeepReprentationAttack)
mlconfig.register(ue_pgd.UEPGD)