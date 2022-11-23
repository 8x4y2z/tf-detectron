# -*- coding: utf-8 -*-

import torch
from torch import nn

from detectron2.layers.wrappers import Conv2d
from detectron2.config.config import configurable


class CenternetHeads(nn.Module):


    @configurable
    def __init__(self,in_channels, nheads_in, hm_heads,rg_heads,wh_heads):
        self.heatmap_layer = nn.Sequential(
            Conv2d(in_channels,nheads_in,kernel_size=3,stride=1,padding="same"),
            nn.ReLU(),
            Conv2d(nheads_in,hm_heads,kernel_size=1,stride=1,padding="same")
        )

        self.reg_layer = nn.Sequential(
            Conv2d(in_channels,nheads_in, kernel_size=(3, 3), strides=1, padding="same"),
            nn.ReLU(),
            Conv2d(nheads_in,rg_heads, kernel_size=(1, 1), strides=1, padding="same")
        )
        self.wh_layer = nn.Sequential(
            Conv2d(in_channels,nheads_in, kernel_size=(3, 3), strides=1, padding="same"),
            nn.ReLU(),
            Conv2d(nheads_in,wh_heads, kernel_size=(1, 1), strides=1, padding="same")
        )


    def forward(self,x):
        heatmap = self.heatmap_layer(x)
        reg = self.reg_layer(x)
        wh = self.wh_layer(x)

        return torch.concat([heatmap, reg, wh],1)


    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels":None,
            "nheads_in":None,
            "hm_heads":None,
            "rg_heads":None,
            "wh_heads":None,
        }
