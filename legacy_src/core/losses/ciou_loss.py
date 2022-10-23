# -*- coding: utf-8 -*-

import math
import tensorflow as tf

def ciou_loss(
    boxes1: tf.Tensor,
    boxes2: tf.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> tf.Tensor:
    """
    Complete Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
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

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    # assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    # assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = tf.maximum(x1, x1g)
    ykis1 = tf.maximum(y1, y1g)
    xkis2 = tf.minimum(x2, x2g)
    ykis2 = tf.minimum(y2, y2g)

    intsct = tf.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # smallest enclosing box
    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # width and height of boxes
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    v = (4 / (math.pi**2)) * tf.math.pow((tf.atan(w_gt / h_gt) - tf.atan(w_pred / h_pred)), 2)
    # with torch.no_grad():
        # alpha = v / (1 - iou + v + eps)
    alpha = tf.stop_gradient(v / (1 - iou + v + eps))

    # Eqn. (10)
    loss = 1 - iou + (distance / diag_len) + alpha * v
    if reduction == "mean":
        loss = tf.math.reduce_mean(loss) if loss.numpy().size > 0 else 0.0 * tf.reduce_sum(loss)
    elif reduction == "sum":
        loss = tf.math.reduce_sum(loss)

    return loss
