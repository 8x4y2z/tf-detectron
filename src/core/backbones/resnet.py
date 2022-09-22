# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPool2D

from src.core.backbones.backbone import BackBone
from src.core.norms import get_norm

class Stem(tf.keras.layers.Layer):
    def __init__(self,filters=64,kernal_size=7,norm="batch_norm"):
        "docstring"
        super(Stem, self).__init__()
        self.conv1 = Conv2D(filters=filters,
                            kernal_size=(7,7),
                            strides=2,
                            padding="same",
                            )
        if norm is not None:
            self.norm = get_norm(norm)

        self.pool1 = MaxPool2D(pool_size=(3, 3),
                               strides=2,
                               padding="same")

    def __call__(self,x,training=None,**kwargs):
        """
        """
        x = self.conv1(x)
        if hasattr(self,"norm"):
            x = self.norm(x,training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        return x


class ResNet(BackBone):
    """
    Implements :paper: 'Resnet'
    """
    def __init__(self, stem, freeze_at=0):
        super(ResNet, self).__init__()
        self.stem = stem

    def __call__(self):
        """
        """

def build_resnet():
    """
    """
