# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPool2D,GlobalAveragePooling2D,Dense

from src.core.backbones.backbone import BackBone
from . import BACKBONE_REGISTRY
from src.core.norms import get_norm

RESNET_BLOCKS = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}

class Stem(tf.keras.layers.Layer):
    def __init__(self,filters=64,kernal_size=7,norm="batch_norm"):
        "docstring"
        super(Stem, self).__init__()
        self.conv1 = Conv2D(filters=filters,
                            kernal_size=kernal_size,
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

class BasicBlock(tf.keras.layers.Layer):
    """
    Basic Residual Block for Resnet18 and Resnet 34
    """
    def __init__(self,filter_num, stride=1,norm="batch_norm"):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = get_norm(norm)
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = get_norm(norm)
        if stride != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filter_num,
                                       kernel_size=(1, 1),
                                       strides=stride,
                                       use_bias=False),
                get_norm(norm),
            ])
        else:
            self.downsample = tf.keras.layers.Lambda(lambda x: x)


    def call(self, inputs, training=None):
        residual = self.downsample(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output





class BottleNeckBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1,norm="batch_norm"):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = get_norm(norm)
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = get_norm(norm)
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = get_norm(norm)
        self.downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=(1, 1),
                                   strides=stride,
                                   use_bias=False),
            get_norm(norm)
        ])

    def call(self, inputs, training=None):
        residual = self.downsample(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output





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
        x = self.stem(x,training=training)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x,training=training)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x,training=training)
            # x = torch.flatten(x, 1)
            x = self.linear(x,training=training)
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


    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.

        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.

        Returns:
            list[list[layers]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = RESNET_BLOCKS[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleNeckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, o) in zip(num_blocks_per_stage, [1, 2, 2, 2],  out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_layer(
                    block_class=block_class,
                    nlayers=n,
                    filters=o,
                    stride=s
                    **kwargs,
                )
            )
        return ret


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(config):
    """
    Create ResNet from config
    """
    norm = config.MODEL.RESNETS.NORM
    stem = Stem(
        config.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = config.MODEL.BACKBONE.FREEZE_AT
    out_features        = config.MODEL.RESNETS.OUT_FEATURES
    depth               = config.MODEL.RESNETS.DEPTH
    num_groups          = config.MODEL.RESNETS.NUM_GROUPS
    out_channels        = config.MODEL.RESNETS.RES2_OUT_CHANNELS
    # fmt: on

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    if depth in [18, 34]:
        block_class = BasicBlock
    else:
        block_class = BottleNeckBlock

    stages = ResNet.make_default_stages(depth,block_class)

    return ResNet(stem, stages, out_layers=out_features, freeze_at=freeze_at)
