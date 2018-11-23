from abc import ABC, abstractmethod

class GraphLayer(ABC):
    """A GraphLayer is a function that transforms a set of node embeddings"""
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn_cell_impl.py#L174
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/engine/base_layer.py#L71

    def __init__(self, layer_params):
        self.layer_params = layer_params

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Abstract Method")
