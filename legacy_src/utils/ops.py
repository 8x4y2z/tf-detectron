# -*- coding: utf-8 -*-

from typing import List

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
