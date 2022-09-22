
from abc import ABCMeta, abstractmethod
import tensorflow as tf

tf.keras.layers.Layer

__all__ = ["BackBone"]

class BackBone(tf.keras.layers.Layer,metaclass=ABCMeta):
    """
    ABC class for all backbones
    """

    @abstractmethod
    def __call__(self):
        """
        """
