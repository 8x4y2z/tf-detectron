# -*- coding: utf-8 -*-

from .smooth_l1_loss import smooth_l1_loss
from .giou_loss import giou_loss
from .diou_loss import diou_loss
from .ciou_loss import ciou_loss

__all__ = ["smooth_l1_loss", "giou_loss","diou_loss","ciou_loss"]
