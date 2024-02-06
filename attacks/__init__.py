#!/usr/bin/python
# author eson
from torchattacks import PGD
import mlconfig
from . import ue_pgd,mix_ce_pgd,drm
mlconfig.register(PGD)
mlconfig.register(ue_pgd.UEPGD)
mlconfig.register(ue_pgd.EOTUEPGD)
mlconfig.register(mix_ce_pgd.MixCEPGD)
mlconfig.register(drm.DeepRepresentationAttack)