from abc import ABC, abstractmethod
from copy import copy

class GraphLayer(ABC):
    """A GraphLayer is a function that transforms a set of node embeddings"""
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn_cell_impl.py#L174
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/engine/base_layer.py#L71

    def __init__(self, layer_params, network_params, name):
        self.layer_params = layer_params
        self.network_params = network_params
        self.name = name

    def clone(self, name=None):
        clone = copy(self)
        if name is None:
            clone.name += '_copy'
        else:
            clone.name = name
        return clone

    def get_node_embeds(layer_input_embeds):
        # input is [num_nodes, max_degree, vector dim]
        end = self.layer_params.node_embed_size,
        sl = slice(0, end, 1)
        return layer_input_embeds[:, :, sl]

    def get_node_label_embeds(layer_input_embeds):
        # input is [num_nodes, max_degree, vector dim]
        start = self.layer_params.node_embed_size
        end = start + self.layer_params.node_label_embed_size
        sl = slice(start, end, 1)
        return layer_input_embeds[:, :, sl]

    def get_edge_label_embeds(layer_input_embeds):
        # input is [num_nodes, max_degree, vector dim]
        start = tf.shape(layer_input_embeds)[-1] - \
            self.layer_params.edge_label_size
        sl = slice(start, None, 1)
        return layer_input_embeds[:, :, sl]

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Abstract Method")
