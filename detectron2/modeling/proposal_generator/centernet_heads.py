# -*- coding: utf-8 -*-

from typing import Dict
import torch
from torch import nn
from detectron2.layers.shape_spec import ShapeSpec

from detectron2.layers.wrappers import Conv2d, ConvTranspose2d, BatchNorm2d
from detectron2.config.config import configurable

from .build import PROPOSAL_GENERATOR_REGISTRY

@PROPOSAL_GENERATOR_REGISTRY.register()
class CenternetHeads(nn.Module):


    @configurable
    def __init__(self,nlayers, in_channels,nheads_in, hm_heads,rg_heads,wh_heads):
        super(CenternetHeads, self).__init__()
        self.transpose_layers = []
        for _ in range(nlayers-1):
            self.transpose_layers.append(
                nn.Sequential(
                    ConvTranspose2d(in_channels,in_channels,kernel_size=(4,4),stride=2,padding=1),
                    BatchNorm2d(in_channels)
                )
            )
        self.heatmap_layer = nn.Sequential(
            Conv2d(in_channels,nheads_in,kernel_size=3,stride=1,padding="same"),
            nn.ReLU(),
            Conv2d(nheads_in,hm_heads,kernel_size=1,stride=1,padding="same")
        )

        self.reg_layer = nn.Sequential(
            Conv2d(in_channels,nheads_in, kernel_size=(3, 3), stride=1, padding="same"),
            nn.ReLU(),
            Conv2d(nheads_in,rg_heads, kernel_size=(1, 1), stride=1, padding="same")
        )
        self.wh_layer = nn.Sequential(
            Conv2d(in_channels,nheads_in, kernel_size=(3, 3), stride=1, padding="same"),
            nn.ReLU(),
            Conv2d(nheads_in,wh_heads, kernel_size=(1, 1), stride=1, padding="same")
        )


    def forward(self,x):
        new_ten = None
        for i in range(len(x)):
            old_ten = (x[i]+new_ten) if new_ten is not None else x[i]
            if i < len(x)-1:
                new_ten = self.transpose_layers[i](old_ten,output_size = x[i+1].size())

        heatmap = self.heatmap_layer(old_ten)
        reg = self.reg_layer(old_ten)
        wh = self.wh_layer(old_ten)

        return torch.concat([heatmap, reg, wh],1)


    @classmethod
    def from_config(cls, cfg,input_shape:Dict[str,ShapeSpec]):
        return {
            "nlayers":len(input_shape),
            "in_channels": next(iter(input_shape.values())).channels,
            "nheads_in":cfg.MODEL.PROPOSAL_GENERATOR.NHEADS,
            "hm_heads":cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "rg_heads":2,
            "wh_heads":2,
        }
