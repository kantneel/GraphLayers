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

    def get_node_ids(self, layer_inputs):
        if tf.rank(layer_inputs) is 3:
            return layer_inputs[:, :, 0]
        elif tf.rank(layer_inputs) is 4:
            return layer_inputs[:, :, :, 0]
        else:
            raise Exception('bad input')

    def get_node_label_ids(self, layer_inputs):
        if tf.rank(layer_inputs) is 3:
            return layer_inputs[:, :, 1]
        elif tf.rank(layer_inputs) is 4:
            return layer_inputs[:, :, :, 1]
        else:
            raise Exception('bad input')

    def get_edge_label_ids(self, layer_inputs):
        if tf.rank(layer_inputs) is 3:
            return layer_inputs[:, :, 2]
        elif tf.rank(layer_inputs) is 4:
            return layer_inputs[:, :, :, 2]
        else:
            raise Exception('bad input')

    @abstractmethod
    def create_node_label_embeds(self):
        # try self.create_default_node_label_embeds()
        raise NotImplementedError("Abstract Method")

    def create_default_node_label_embeds(self):
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name):
            self.node_label_embeds = tf.Variable(
                initializer([
                    self.network_params.num_node_labels,
                    self.layer_params.node_label_embed_size]),
                name='node_label_embeds')

    @abstractmethod
    def create_edge_label_embeds(self):
        # try self.create_default_edge_label_embeds()
        raise NotImplementedError("Abstract Method")

    def create_default_edge_label_embeds(self):
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name):
            self.edge_label_embeds = tf.Variable(
                initializer([
                    self.network_params.num_edge_labels,
                    self.layer_params.edge_label_embed_size]),
                name='edge_label_embeds')

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Abstract Method")

    def __str__(self):
        pass

    def __repr__(self):
        pass

