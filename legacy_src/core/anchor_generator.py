# -*- coding: utf-8 -*-

import math
from typing import List
from collections.abc import Sequence
import tensorflow as tf

from src.config.config import Configurable
from src.utils.registry import Registry
from src.structures import ShapeSpec, Boxes


ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
ANCHOR_GENERATOR_REGISTRY.__doc__ = """
Registry for modules that creates object detection anchors for feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
"""

class BufferList(tf.keras.layers.Layer):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        _initializer = lambda buf:tf.constant_initializer(buf.numpy())
        self._buffers={}
        for i, buffer in enumerate(buffers):
            buf=self.add_weight(
                name=str(i),
                shape=buffer.shape,
                dtype=buffer.dtype,
                initializer=_initializer(buffer),
                trainable=False,
            )
            self._buffers[str(i)] = buf

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())



def _create_grid_offsets(
    size: List[int], stride: int, offset: float):
    grid_height, grid_width = size
    shifts_x = tf.arange(offset * stride, grid_width * stride, step=stride, dtype=tf.float32)
    shifts_y = tf.arange(offset * stride, grid_height * stride, step=stride, dtype=tf.float32)

    shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x,indexing="ij")
    shift_x = tf.reshape(shift_x,-1)
    shift_y = tf.reshape(shift_y,-1)
    return shift_x, shift_y

@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(tf.keras.layers.Layer, metaclass=Configurable):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    """

    box_dim: int = 4
    """
    the dimension of each anchor box.
    """

    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        """
        This interface is experimental.

        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        kargs= {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
        }
        return cls(**kargs)

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        buffers: List[tf.Tensor] = [x for x in self.cell_anchors._buffers.values()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset)
            shifts = tf.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return tf.convert_to_tensor(anchors)

    def call(self, features: List[tf.Tensor]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]

def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(
        params, Sequence
    ), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], Sequence):  # params is list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params

def build_anchor_generator(cfg, input_shape):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)
