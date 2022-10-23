# -*- coding: utf-8 -*-

from src.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a TF Model
"""


def build_model(config):
    """
    Build the whole model architecture, defined by ``config.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``config``.
    """
    meta_arch = config.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(config)
    return model
