# -*- coding: utf-8 -*-

from typing import List, Tuple

import tensorflow as tf
import numpy as np

__all__ = ["nonzero_tuple","scatter_tf","cat"]

def nonzero_tuple(x):
    zeros = tf.constant(0,dtype=x.dtype)
    out = tf.where(tf.math.not_equal(x,zeros))
    return out[:,0],out[:,1]

def scatter_tf(out,index,src,dim):
    # import pdb; pdb.set_trace()
    tf.debugging.assert_integer(index,"The values of index must be integers")
    if out.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= out.shape.rank or dim < -out.shape.rank:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
        dim = out.shape.rank + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    # out_xsection_shape = out.shape[:dim] + out.shape[dim + 1:]
    # if idx_xsection_shape != out_xsection_shape:
    #     raise ValueError("Except for dimension " + str(dim) +
    #                      ", all dimensions of index and output should be the same size")
    if tf.math.reduce_any(index >= out.shape[dim]) or tf.math.reduce_any(index < 0):
        raise IndexError("The values of index must be between 0 and (out.shape[dim] -1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return slc

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in out
    idx = [[*np.indices(idx_xsection_shape.as_list()).reshape(index.ndim - 1, -1),
            index.numpy()[make_slice(index.numpy(), dim, i)].reshape(1, -1)[0]] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())
    new_idx = tf.stack(idx,axis=1)
    if not tf.experimental.numpy.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        # src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        # if idx_xsection_shape != src_xsection_shape:
        #     raise ValueError("Except for dimension " +
        #                      str(dim) + ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))

        src_idx = tf.cast(tf.stack(src_idx,axis=1),out.dtype)

        new_src = tf.gather_nd(src,src_idx)
        out = tf.tensor_scatter_nd_update(out,new_idx,new_src)

    else:
        new_src = tf.repeat(tf.convert_to_tensor(src),new_idx.shape[0])
        new_src = tf.cast(new_src,out.dtype)
        out = tf.tensor_scatter_nd_update(out,new_idx,new_src)

    return out


def cat(tensors: List[tf.Tensor], axis: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return tf.concat(tensors, axis=axis)

def find_top_rpn_proposals(
    proposals: List[tf.Tensor],
    pred_objectness_logits: List[tf.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    # device = (
    #     proposals[0].device
    #     if torch.jit.is_scripting()
    #     else ("cpu" if torch.jit.is_tracing() else proposals[0].device)
    # )

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = tf.range(num_images)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, tf.Tensor):  # it's a tensor in tracing
            num_proposals_i = tf.clamp(Hi_Wi_A,tf.math.reduce_min(Hi_Wi_A) ,pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        topk_scores_i, topk_idx = tf.math.top_k(logits_i,num_proposals_i)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            move_device_like(
                torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device),
                proposals[0],
            )
        )

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results
