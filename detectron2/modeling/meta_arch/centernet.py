# -*- coding: utf-8 -*-
from typing import Tuple, Optional, List, Dict

import torch
from torch import nn, topk
import torch.nn.functional as F

from detectron2.config.config import configurable
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.layers import move_device_like, gaussian_radius, draw_umich_gaussian
from detectron2.utils.events import get_event_storage
from detectron2.structures import Instances, ImageList, Boxes
from detectron2.layers import modified_focal_loss, reg1loss
from detectron2.layers import gather_feat_alt, gather_feat
from detectron2.layers.losses import _transpose_and_gather_feat
from ..postprocessing import detector_postprocess
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
                 downsampling:int,
                 focal_loss_alpha:float,
                 focal_loss_beta:float,
                 thresh:float,
                 vis_period: int = 0,
                 ):
        super(Centernet, self).__init__()
        self.backbone = backbone
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
        self.downsampling = downsampling
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_beta = focal_loss_beta
        self.thresh = thresh



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
            "max_boxes": cfg.TEST.DETECTIONS_PER_IMAGE,
            "hm_weight": cfg.MODEL.PROPOSAL_GENERATOR.HM_WEIGHT,
            "reg_weight": cfg.MODEL.PROPOSAL_GENERATOR.REG_WEIGHT,
            "wh_weight": cfg.MODEL.PROPOSAL_GENERATOR.WH_WEIGHT,
            "downsampling": cfg.MODEL.CENTERNET.DOWN_SAMPLING,
            "focal_loss_alpha": cfg.MODEL.CENTERNET.FOCAL_LOSS_ALPHA,
            "focal_loss_beta": cfg.MODEL.CENTERNET.FOCAL_LOSS_BETA,
            "thresh": cfg.MODEL.CENTERNET.THRESHOLD
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

        losses = self._combined_losses(out_features, gt_instances, images.tensor.shape[-2:])

        return losses

    def _combined_losses(self,out_features, gt_instances,img_shp):
        gt_hm, gt_reg, gt_wh, gt_reg_mask, gt_indices = self._decode(self.nclasses,gt_instances,img_shp)
        heatmap, reg, wh = torch.split(out_features, [self.nclasses, 2, 2], 1)
        heatmap = torch.clamp(torch.sigmoid(heatmap), min=1e-4, max=1.0 - 1e-4)
        hm_loss = modified_focal_loss(heatmap,gt_hm, self.focal_loss_alpha, self.focal_loss_beta)
        reg_loss = reg1loss(reg, gt_reg_mask, gt_indices, gt_reg)
        wh_loss = reg1loss(wh,gt_reg_mask,gt_indices,gt_wh)

        return {
            "hm_loss":hm_loss,
            "reg_loss":reg_loss,
            "wh_loss":wh_loss,
            "hm_weight": self.hm_weight,
            "reg_weight": self.reg_weight,
            "wh_weight": self.wh_weight
        }

    def _decode(self,nclasses:int,gt_instances:List[Instances], img_shp):
        features_shape = tuple(x//self.downsampling for x in img_shp)
        gt_hm = torch.zeros(
            (len(gt_instances),nclasses,*features_shape),
            dtype=torch.float32,device=gt_instances[0].gt_boxes.device
        )
        gt_reg = torch.zeros(
            (len(gt_instances),self.max_boxes,2),
            dtype=torch.float32,device=gt_instances[0].gt_boxes.device
        )
        gt_wh = torch.zeros(
            (len(gt_instances),  self.max_boxes,2),
            dtype=torch.float32,device=gt_instances[0].gt_boxes.device
                            )
        gt_reg_mask = torch.zeros(
            (len(gt_instances), self.max_boxes),
            dtype=torch.float32,device=gt_instances[0].gt_boxes.device
        )
        gt_indices = torch.zeros(
            (len(gt_instances), self.max_boxes),
            dtype=torch.int64,device=gt_instances[0].gt_boxes.device
        )

        for i, instance in enumerate(gt_instances):
            # label = label[label[:, 4] != -1]
            hm, reg, wh, reg_mask, ind = self._decode_label(instance,img_shp)
            gt_hm[i, :, :, :] = hm
            gt_reg[i, :, :] = reg
            gt_wh[i, :, :] = wh
            gt_reg_mask[i, :] = reg_mask
            gt_indices[i, :] = ind


        return gt_hm, gt_reg, gt_wh, gt_reg_mask, gt_indices

    def _decode_label(self, instance:Instances, img_shp):

        features_shape = tuple(x//self.downsampling for x in img_shp)
        hm = torch.zeros(
            (self.nclasses, *(features_shape)),
            dtype=torch.float32, device=instance.gt_boxes.device)
        reg = torch.zeros((self.max_boxes,2), dtype=torch.float32,
                          device=instance.gt_boxes.device)
        wh = torch.zeros((self.max_boxes,2),
                         dtype=torch.float32,device=instance.gt_boxes.device)
        reg_mask = torch.zeros((self.max_boxes),
                               dtype=torch.float32, device=instance.gt_boxes.device)
        ind = torch.zeros((self.max_boxes),
                          dtype=torch.float32, device=instance.gt_boxes.device)
        instance.gt_boxes.scale(1/self.downsampling, 1/self.downsampling)
        for i,box in enumerate(instance.gt_boxes):
            xmin,ymin,xmax,ymax = box
            class_id = instance.gt_classes[i]
            h, w = int(ymax - ymin), int(xmax - xmin)
            radius = gaussian_radius((h, w))
            radius = max(0, int(radius))
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = torch.tensor([ctr_x, ctr_y],
                                        dtype=torch.float32, device=instance.gt_boxes.device)
            center_point_int = center_point.to(torch.int32)
            draw_umich_gaussian(hm[class_id], center_point_int, radius)
            reg[i] = center_point - center_point_int
            wh[i] = torch.tensor((1. * w, 1. * h)).to(wh)
            reg_mask[i] = 1
            ind[i] = center_point_int[1] * features_shape[1] + center_point_int[0]
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

        odims = images.tensor.shape[-2:]

        if detected_instances is None:
            if self.proposal_generator is not None:
                out_features = self.proposal_generator(features)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results = self._interpret(out_features,odims)
        else:
            raise RuntimeError("Unknow error occured")

        instances = []
        for result,img_size in zip(results,images.image_sizes):
            boxes, scores, classes = torch.split(result,[4,1,1],-1)
            kwargs = {"pred_boxes":Boxes(boxes),
                      "scores":scores.reshape(-1),
                      "pred_classes":classes.reshape(-1)
                      }
            instance = Instances(img_size,**kwargs)
            instances.append(instance)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return Centernet._postprocess(instances, batched_inputs, images.image_sizes)
        return results

    @staticmethod
    def quick_viz(binput,output,annos,dp="testinfer.jpg"):
        import numpy as np,cv2
        inp_img = np.ascontiguousarray(np.transpose(binput["image"].cpu().numpy(),[1,2,0]))
        for bo in output:
            _ = cv2.rectangle(inp_img,
                              tuple(int(a) for a in bo[:2]),
                              tuple(int(a) for a in bo[2:]),
                              (255,0,1),2)

        select = [anno for anno in annos if anno["image_id"] == binput["image_id"]]
        for anno in select:
            x1,y1,w,h = anno["bbox"]
            wf = inp_img.shape[1]/binput["width"]
            hf = inp_img.shape[0]/binput["height"]
            a1,b1,a2,b2 = int(x1*wf), int(y1*hf),int((x1+w)*wf),int((y1+h)*hf)
            _ = cv2.rectangle(inp_img,(a1,b1),(a2,b2),(0,255,0),1)

        cv2.imwrite(dp,inp_img)



    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]).to(torch.float32) for x in batched_inputs]
        images = [y/255.0 for y in images] # Squash b/w 0 and 1
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

    def _interpret(self,out_features,img_shape):
        detections = centernet_decode(
            out_features,
            img_shape,
            self.nclasses,
            self.max_boxes,
            self.downsampling,
            self.thresh)
        return detections




def centernet_decode(
        out_features,
        img_shape,
        nclasses,
        top_k,
        down_sampling,
        thresh
):
    heatmap, reg, wh = torch.split(out_features, [nclasses, 2, 2], 1)
    heatmap = torch.sigmoid(heatmap)
    heatmap = centernet_nms(heatmap)
    batch_size = out_features.shape[0]
    scores, inds, clses, ys, xs = centernet_topk(heatmap,top_k)

    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch_size, top_k, 2)
    xs = xs.view(batch_size, top_k, 1) + reg[:, :, 0:1]
    ys = ys.view(batch_size, top_k, 1) + reg[:, :, 1:2]

    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch_size, top_k, 2)

    clses  = clses.view(batch_size, top_k, 1).float()
    scores = scores.view(batch_size, top_k, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)


    # reg = gather_feat_alt(reg,inds)
    # xs = torch.reshape(xs,(batch_size,top_k,1)) + reg[:,:,0:1]
    # ys = torch.reshape(ys,(batch_size, top_k, 1)) + reg[:, :, 1:2]
    # wh = gather_feat_alt(wh,inds)

    # classes = torch.reshape(clses, (batch_size, top_k, 1)).to(torch.float32)
    # scores = torch.reshape(scores, (batch_size, top_k, 1))
    # bboxes = torch.cat([xs - wh[..., 0:1] / 2,
    #                            ys - wh[..., 1:2] / 2,
    #                            xs + wh[..., 0:1] / 2,
    #                            ys + wh[..., 1:2] / 2], 2)
    # detections = torch.cat([bboxes, scores, classes], 2)

    return centernet_map_to_original(detections,img_shape,down_sampling,thresh)


def centernet_topk(heatmap,top_k):
    B, C, H, W = heatmap.shape
    scores = torch.reshape(heatmap, (B,C,-1))
    topk_scores, topk_inds = torch.topk(scores,top_k, sorted=True)


    topk_inds = topk_inds % (H * W)
    topk_ys   = (topk_inds / W).int().float()
    topk_xs   = (topk_inds % W).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), top_k)
    topk_clses = (topk_ind / top_k).int()
    topk_inds = gather_feat(
        topk_inds.view(B, -1, 1), topk_ind).view(B, top_k)
    topk_ys = gather_feat(topk_ys.view(B, -1, 1), topk_ind).view(B, top_k)
    topk_xs = gather_feat(topk_xs.view(B, -1, 1), topk_ind).view(B, top_k)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
    
    # topk_clses = topk_inds % C
    # topk_xs = (topk_inds // C % W).to(torch.float32)
    # topk_ys = (topk_inds // C // W).to(torch.float32)
    # topk_inds = (topk_ys * (W) + topk_xs).to(torch.int64)
    # return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def centernet_nms(heatmap, pool_size=3):
    hmax = torch.nn.MaxPool2d(pool_size, stride=1, padding=1)(heatmap)
    keep = (heatmap == hmax).to(torch.float32)
    return hmax * keep


def centernet_map_to_original(detections,original_image_size,downsampling_ratio,score_threshold):
    bboxes, scores, clses = torch.split(detections, [4, 1, 1], 2)
    # bboxes, scores, clses = bboxes[0], scores[0], clses[0]
    # resize_ratio = original_image_size / self.input_image_size
    bboxes[..., 0::2] = bboxes[..., 0::2] * downsampling_ratio
    bboxes[..., 1::2] = bboxes[..., 1::2] * downsampling_ratio
    bboxes[..., 0::2] = torch.clamp(bboxes[..., 0::2], min=0, max=original_image_size[1])
    bboxes[..., 1::2] = torch.clamp(bboxes[..., 1::2], min=0, max=original_image_size[0])
    score_mask = scores >= score_threshold
    bboxes, scores, clses = _numpy_mask(bboxes, torch.tile(score_mask, (1, 4))), _numpy_mask(scores, score_mask), _numpy_mask(clses, score_mask)
    detections = torch.cat([bboxes, scores, clses], -1)
    return detections


def _numpy_mask(a, mask):
    return a[mask].reshape(a.shape[0],-1, a.shape[-1])
