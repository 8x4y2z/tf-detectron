# -*- coding: utf-8 -*-

import tensorflow as tf

from src.core.backbones import build_backbone
from src.core.proposal_generators import build_proposal_generator
from .build import META_ARCH_REGISTRY
from src.core.postprocessing import detector_postprocess
from src.config import Configurable


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(tf.keras.layers.Layer,metaclass=Configurable):
    """Implemets a Generalized RCNN composed of following components:
    1. Feature extractor aka BackBone
    2. RPN
    3. Per-region feature extractor and prediction
    """

    def __init__(self,*,backbone,proposal_generator,roi_heads,pixel_mean,pixel_std):
        """
        :param backbone: is feature extractor
        :param proposal_generator: is RPN
        :param ROI Computation: ROI head performs per region computation
        :param pixel_mean: pixel_std: Per channel mean and std to be considered for normalization
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.pixel_mean = tf.Variable(pixel_mean.reshape(-1,1,1),trainable=False,name="pixel_mean")
        self.pixel_mean = tf.Variable(pixel_std.reshape(-1,1,1),trainable=False,name="pixel_std")

    def call(self,inputs,training=None):
        """
        """
        features = self.backbone(inputs,training=training)
        proposals = self.proposal_generator(inputs,features,training=training)
        outputs = self.roi_heads(inputs,features,proposals,training=training)
        return features,proposals,outputs

    @classmethod
    def from_config(cls,cfg):
        backbone = build_backbone(cfg)
        proposal_gen = build_proposal_generator(cfg, backbone.output_shape())
        roi_heads = build_roi_head(cfg, backbone.output_shape())

        return cls(**{
            "backbone": backbone,
            "proposal_generator": proposal_gen,
            "roi_heads": roi_heads,
            # "input_format": cfg.INPUT.FORMAT,
            # "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        })


@META_ARCH_REGISTRY.register()
class ProposalNetwork(tf.keras.layers.Layer):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(
        self,
            backbone,
            proposal_generator,
            pixel_mean,
            pixel_std
    ):
        """
        :param backbone: is feature extractor
        :param proposal_generator: is RPN
        :param ROI Computation: ROI head performs per region computation
        :param pixel_mean: pixel_std: Per channel mean and std to be considered for normalization
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator

        self.pixel_mean = tf.Variable(pixel_mean.reshape(-1,1,1),trainable=False,name="pixel_mean")
        self.pixel_mean = tf.Variable(pixel_std.reshape(-1,1,1),trainable=False,name="pixel_std")


    def forward(self, batched_inputs,training=None):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """

        features = self.backbone(batched_inputs,training=training)
        proposals = self.proposal_generator(batched_inputs,features)

        processed_results = []
        for results_per_image, input_per_image in zip(proposals, batched_inputs):
            height = input_per_image
            width = input_per_image
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
