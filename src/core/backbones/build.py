#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.utils.registry import Registry


BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry of available backbones
"""

def build_backbone(config):
    """Build Backbone from config
    """
    backbone_name = config.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(config)
    return backbone
