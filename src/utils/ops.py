# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = ["nonzero_tuple"]

def nonzero_tuple(x):
    zeros = tf.constant(0,dtype=x.dtype)
    out = tf.where(tf.math.not_equal(x,zeros))
    return out[:,0],out[:,1]
