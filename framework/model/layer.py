from abc import ABC, abstractmethod
from copy import copy

class GraphLayer(ABC):
    """A GraphLayer is a function that transforms a set of node embeddings"""
    def __init__(self, layer_params, network_params, name):
        self.layer_params = layer_params
        self.network_params = network_params
        self.name = name
        self.input_config = InputConfig.default()

    def clone(self, name=None):
        clone = copy(self)
        if name is None:
            clone.name += '_copy'
        else:
            clone.name = name
        return clone

    def create_weights(self):
        pass

    def reset(self):
        pass

    def get_node_ids(self, layer_inputs):
        if tf.rank(layer_inputs) is 3:
            return layer_inputs[:, : 0]
        elif tf.rank(layer_inputs) is 4:
            return layer_inputs[:, :, :, 0]
        else:
            raise Exception('bad input')

    def get_node_label_ids(self, layer_inputs):
        if tf.rank(layer_inputs) is 3:
            return layer_inputs[:, : 1]
        elif tf.rank(layer_inputs) is 4:
            return layer_inputs[:, :, :, 1]
        else:
            raise Exception('bad input')

    def get_edge_label_ids(self, layer_inputs):
        if tf.rank(layer_inputs) is 3:
            return layer_inputs[:, : 2]
        elif tf.rank(layer_inputs) is 4:
            return layer_inputs[:, :, :, 2]
        else:
            raise Exception('bad input')

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Abstract Method")
