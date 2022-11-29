# -*- coding: utf-8 -*-

from typing import Dict
import torch
from torch import nn
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.layers.wrappers import Conv2d,  BatchNorm2d
from detectron2.config.config import configurable

from .build import PROPOSAL_GENERATOR_REGISTRY

@PROPOSAL_GENERATOR_REGISTRY.register()
class CenternetHeads(nn.Module):


    @configurable
    def __init__(self,*,nlayers, in_channels,nheads_in, hm_heads,rg_heads,wh_heads):
        super(CenternetHeads, self).__init__()
        assert nlayers == 5, "Currently only supports p2,p3,p4,p5,p6"
        self.bn0 = BatchNorm2d(in_channels)
        self.bn1 = BatchNorm2d(in_channels)
        self.bn2 = BatchNorm2d(in_channels)
        self.bn3 = BatchNorm2d(in_channels)

        self.tp0 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=(1,1), stride=(2,2), padding=(0,0)
        )
        self.tp1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0)
        )
        self.tp2 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0)
        )
        self.tp3 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0)
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
        out = self.tp0(x["p6"], output_size = x["p5"].size())
        out = self.bn0(out)
        # out = F.interpolate(out,(25,34))
        x["p5"] = torch.add(x["p5"], out)

        out = self.tp1(x["p5"],output_size = x["p4"].size())
        out = self.bn1(out)
        # out = F.interpolate(out,(50,68))
        x["p4"] += out

        out = self.tp2(x["p4"],output_size = x["p3"].size())
        out = self.bn2(out)
        # out = F.interpolate(out,(100,136))
        x["p3"] += out

        out = self.tp3(x["p3"],output_size = x["p2"].size())
        out = self.bn3(out)
        # out = F.interpolate(out,(200,272))
        x["p2"] += out

        heatmap = self.heatmap_layer(x["p2"])
        reg = self.reg_layer(x["p2"])
        wh = self.wh_layer(x["p2"])
        return torch.cat([heatmap, reg, wh],1)


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
