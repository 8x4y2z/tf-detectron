# -*- coding: utf-8 -*-

from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.layers.wrappers import Conv2d,  BatchNorm2d
from detectron2.config.config import configurable

from .build import PROPOSAL_GENERATOR_REGISTRY

@PROPOSAL_GENERATOR_REGISTRY.register()
class CenternetHeads(nn.Module):


    @configurable
    def __init__(self,*,nlayers, in_channels,nheads_in, hm_heads,rg_heads,wh_heads):
        super(CenternetHeads, self).__init__()
        self.transpose_layers = []
        self.bnlayers = nn.Sequential(
           *[BatchNorm2d(in_channels) for _ in range(nlayers-1)]
        )
        for _ in range(nlayers-2):
            self.transpose_layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        in_channels,
                        kernel_size=(2,2),
                        stride=2,
                        padding=(0,0),
                        dilation=1
                        # output_padding=1,
                    )
            )

        self.transpose_layers = nn.Sequential(*self.transpose_layers)
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
        # out = self.transpose_layers[0](x["p6"])
        # out = self.bnlayers[0](out)
        # out = F.interpolate(out,(25,34))
        # x["p5"] += out

        out = self.transpose_layers[0](x["p5"],output_size = x["p4"].size())
        out = self.bnlayers[0](out)
        # out = F.interpolate(out,(50,68))
        x["p4"] += out

        out = self.transpose_layers[1](x["p4"],output_size = x["p3"].size())
        out = self.bnlayers[1](out)
        # out = F.interpolate(out,(100,136))
        x["p3"] += out

        out = self.transpose_layers[2](x["p3"],output_size = x["p2"].size())
        out = self.bnlayers[2](out)
        # out = F.interpolate(out,(200,272))
        x["p2"] += out

        # out = self.transpose_layers[0](x["p6"])
        # out = self.bnlayers[0](out)
        # out = F.interpolate(out,(25,34))
        # x["p5"] += out

        # ps = ("p6","p5","p4","p3","p2")
        # for i,p in enumerate(ps):
        #     if i < 4:
        #         out = self.transpose_layers[i](x[p])
        #         out = self.bnlayers[i](out)
        #         out = F.interpolate(out,x[ps[i+1]].shape[-2:])
        #         x[ps[i+1]] += out

        heatmap = self.heatmap_layer(x["p2"])
        reg = self.reg_layer(x["p2"])
        wh = self.wh_layer(x["p2"])
        return torch.cat([heatmap, reg, wh],1)

    # @torch.no_grad()
    # def _adjust_dim(self,t1:torch.Tensor,t2:torch.Tensor):
    #     r_t1,c_t1 = t1.shape[-2:]
    #     r_t2,c_t2 = t2.shape[-2:]
    #     max_row = max(r_t1,r_t2) if r_t1 != r_t2 else None
    #     max_col = max(c_t1,c_t2) if c_t1 != c_t2 else None

    #     if max_row is not None:
    #         to_add = t2 if max_row == r_t1 else t1
    #         while max_row - min(r_t1,r_t2)> 0:
    #             to_add = torch.cat(
    #                 (
    #                     to_add,
    #                  torch.zeros((*to_add.shape[:2],1,to_add.shape[-1]),device=to_add.device)
    #                 ),
    #                 -2
    #             )
    #             max_row -= 1
    #         if r_t1 > r_t2:
    #             t2 = to_add
    #         else:
    #             t1 = to_add

    #     if max_col is not None:
    #         to_add = t2 if max_col == c_t1 else t1
    #         while max_col - min(c_t1,c_t2) > 0:
    #             to_add = torch.cat(
    #                 (
    #                     to_add,
    #                  torch.zeros((*to_add.shape[:3],1),device=to_add.device)
    #                 ),
    #                 -2
    #             )
    #             max_col -=1

    #         if c_t1 > c_t2:
    #             t2 = to_add
    #         else:
    #             t1 = to_add
    #     return t1,t2


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
