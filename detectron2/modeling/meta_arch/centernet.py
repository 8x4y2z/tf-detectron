# -*- coding: utf-8 -*-
from typing import Tuple, Optional, List, Dict

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config.config import configurable
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.layers import move_device_like, gaussian_radius, draw_umich_gaussian
from detectron2.utils.events import get_event_storage
from detectron2.structures import Instances, ImageList
from detectron2.layers import modified_focal_loss, reg1loss
from .build import META_ARCH_REGISTRY



@META_ARCH_REGISTRY.register()
class Centernet(nn.Module):

    @configurable
    def __init__(self,*,backbone: Backbone,
                 proposal_generator: nn.Module,
                 pixel_mean: Tuple[float],
                 pixel_std: Tuple[float],
                 input_format: Optional[str] = None,
                 hm_weight: float,
                 reg_weight: float,
                 wh_weight: float,
                 nclasses: int,
                 max_boxes:int,
                 vis_period: int = 0,
                 ):
        super(Centernet, self).__init__()
        self.backbone = backbone,
        self.proposal_generator = proposal_generator
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight
        self.nclasses = nclasses
        self.max_boxes = max_boxes



    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "nclasses": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "max_boxes": cfg.TEST.DETECTIONS_PER_IMAGE
        }


    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        out_features = self.proposal_generator(features)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, out_features)

        losses = self._combined_losses(out_features, gt_instances)

        return losses

    def _combined_losses(self,out_features, gt_instances):
        gt_hm, gt_reg, gt_wh, gt_reg_mask, gt_indices = self._decode_gt(self.nclases,gt_instances, out_features)
        heatmap, reg, wh = torch.split(out_features, [self.nclasses, 2, 2], 1)
        heatmap = torch.clamp(F.sigmoid(heatmap), min=1e-4, max=1.0 - 1e-4)
        hm_loss = modified_focal_loss(heatmap,gt_hm, self.focal_loss_alpha, self.focal_loss_beta)
        reg_loss = reg1loss(reg, gt_reg_mask, gt_indices, gt_reg)
        wh_loss = reg1loss(wh,gt_reg_mask,gt_indices,gt_wh)

        loss = self.hm_weight * hm_loss + self.reg_weight * reg_loss + self.wh_weight * wh_loss
        return loss

    def _decode(self,nclasses:int,gt_instances:List[Instances], out_features: torch.Tensor):
        gt_hm = torch.zeros(
            (len(gt_instances),nclasses,*out_features.shape[-2:]),
            dtype=torch.float32,device=gt_instances[0].device
        )
        gt_reg = torch.zeros(
            (len(gt_instances),2,self.max_boxes),
            dtype=torch.float32,device=gt_instances[0].device
        )
        gt_wh = torch.zeros(
            (len(gt_instances), 2, self.max_boxes),
            dtype=torch.float32,device=gt_instances[0].device
                            )
        gt_reg_mask = torch.zeros(
            (len(gt_instances), self.max_boxes),
            dtype=torch.float32,device=gt_instances[0].device
        )
        gt_indices = torch.zeros(
            (len(gt_instances), self.max_boxes),
            dtype=torch.float32,device=gt_instances[0].device
        )

        for i, instance in enumerate(gt_instances):
            label = label[label[:, 4] != -1]
            hm, reg, wh, reg_mask, ind = self.__decode_label(instance)
            gt_hm[i, :, :, :] = hm
            gt_reg[i, :, :] = reg
            gt_wh[i, :, :] = wh
            gt_reg_mask[i, :] = reg_mask
            gt_indices[i, :] = ind


        return gt_hm, gt_reg, gt_wh, gt_reg_mask, gt_indices

    def _decode_label(self, instance:Instances):
        hm = torch.zeros(
            (self.nclasses, *(tuple(x//self.downsampling for x in instance.image_size))),
            dtype=torch.float32, device=instance.device)
        reg = torch.zeros((2,self.max_boxes), dtype=torch.float32,
                          device=instance.boxes.device)
        wh = torch.zeros((2,self.max_boxes),
                         dtype=torch.float32,device=instance.device)
        reg_mask = torch.zeros((self.max_boxes),
                               dtype=torch.float32, device=instance.device)
        ind = torch.zeros((self.max_boxes),
                          dtype=torch.float32, device=instance.device)
        instance.boxes.scale((1/self.downsampling, 1/self.downsampling))
        for i,box in enumerate(instance.boxes):
            xmin,ymin,xmax,ymax = box
            class_id = instance.class_id[i]
            h, w = int(ymax - ymin), int(xmax - xmin)
            radius = gaussian_radius((h, w))
            radius = max(0, int(radius))
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = torch.tensor([ctr_x, ctr_y], dtype=torch.float32, device=instance.boces.device)
            center_point_int = center_point.to(torch.int32)
            draw_umich_gaussian(hm[:, :, class_id], center_point_int, radius)
            reg[i] = center_point - center_point_int
            wh[i] = 1. * w, 1. * h
            reg_mask[i] = 1
            ind[i] = center_point_int[1] * self.features_shape[1] + center_point_int[0]
        return hm, reg, wh, reg_mask, ind



    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return Centernet._postprocess(results, batched_inputs, images.image_sizes)
        return results


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
