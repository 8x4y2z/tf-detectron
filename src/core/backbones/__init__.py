# -*- coding: utf-8 -*-

from src.utils.registry import Registry
from .backbone import BackBone

__all__ = ["BackBone","build_backbone"]

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry of available backbones
Implemented:
RESNET
========================================
TODO:
FPN
=======================================
"""

def build_backbone(config):
    """Build Backbone from config
    """
    backbone_name = config.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(config)
    return backbone
