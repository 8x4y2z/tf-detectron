# -*- coding: utf-8 -*-

# Any custom Layers

import tensorflow as tf



class CConv2D(tf.keras.layers.Conv2D):
    """Custom Conv2d with optional batch norm if required
    """

    def __init__(self,*args,**kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def call(self, x,*args,**kwargs):

        x = super(CConv2D, self).call(x)
        if self.norm is not None:
            x = self.norm(x,*args,**kwargs)
        if self.activation is not None:
            x = self.activation(x,*args,**kwargs)
        return x
