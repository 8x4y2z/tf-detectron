# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPool2D,GlobalAveragePooling2D,Dense

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

    def call(self,x,training=None):
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
    def __init__(self, stem, stages,nclasses=None,out_layers=None,freeze_at=0):
        """
        :param stem: is stem
        :param stages: Resnet Blocks, each containing several CNN block
        :param nclasses: performs classification if not None
        :param out_layer: layers whose output should be considered as final output
        :freeze_at: Layer till which the model should be frozen
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.nclasses = nclasses
        self.stage_names,self.stages = [],[]
        self._out_feature_channels = {"stem": self.stem.out_channels}

        if out_layers is not None:
            nstages = max([{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_layers])
            stages = stages[:nstages]

        for i,blocks in enumerate(stages):
            stage_name = f"res{i+2}"
            stage = tf.keras.Sequential(blocks)
            self.stage_names.append(stage_name)
            self.stages.append(stage)
            self._out_feature_channels[stage_name]  = blocks[-1].out_channels

        if nclasses is not None:
            self.avgpool = GlobalAveragePooling2D()
            self.linear = Dense(nclasses)
            stage_name = "linear"

        if out_layers is None:
            out_layers = [stage_name]
        self._out_layers = out_layers

        self.freeze(freeze_at)


    @staticmethod
    def make_layer(block_class,nlayers,filters,stride=1):
        """Create a list of layers that forms one ResNet Stage
        """
        block = tf.keras.Sequential()
        block.add(block_class(filters,stride=stride))

        for _ in range(1,nlayers):
            block.add(block_class(filters,stride=1))

        return block



    def call(self,x,training=None):
        """
        """
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_layers:
                outputs["linear"] = x
        return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.trainable = False
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for layer in stage.layers:
                    layer.trainable = False
        return self

    def output_shape(self):
        return {
            name: self._out_feature_channels[name] for name in self._out_layers
        }


def build_resnet():
    """
    """
