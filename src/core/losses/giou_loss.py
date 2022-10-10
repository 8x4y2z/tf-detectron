# -*- coding: utf-8 -*-

import tensorflow as tf

def giou_loss(
    boxes1: tf.Tensor,
    boxes2: tf.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> tf.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630
    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = [tf.squeeze(tt,axis=-1) for tt in tf.split(boxes1,boxes1.shape[-1],axis=-1)]
    x1g, y1g, x2g, y2g = [tf.squeeze(tt,axis=-1) for tt in tf.split(boxes2,boxes2.shape[-1],axis=-1)]

    # assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    # assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = tf.maximum(x1, x1g)
    ykis1 = tf.maximum(y1, y1g)
    xkis2 = tf.minimum(x2, x2g)
    ykis2 = tf.minimum(y2, y2g)

    intsctk = tf.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    if reduction == "mean":
        loss = tf.math.reduce_mean(loss) if loss.numpy().size > 0 else 0.0 * tf.reduce_sum(loss)
    elif reduction == "sum":
        loss = tf.math.reduce_sum(loss)

    return loss
