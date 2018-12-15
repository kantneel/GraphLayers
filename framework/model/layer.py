import sys
sys.path.append('../../')
import tensorflow as tf
from abc import ABC, abstractmethod
from copy import copy
from framework.utils.paramspaces import InputConfig


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

    def get_input_config(self):
        return self.input_config

    def create_weights(self):
        pass

    #def get_node_ids(self, layer_inputs):
    #    if tf.rank(layer_inputs) is 3:
    #        return layer_inputs[:, :, 0]
    #    elif tf.rank(layer_inputs) is 4:
    #        return layer_inputs[:, :, :, 0]
    #    else:
    #        raise Exception('bad input')

    #def get_node_label_ids(self, layer_inputs):
    #    if tf.rank(layer_inputs) is 3:
    #        return layer_inputs[:, :, 1]
    #    elif tf.rank(layer_inputs) is 4:
    #        return layer_inputs[:, :, :, 1]
    #    else:
    #        raise Exception('bad input')

    #def get_edge_label_ids(self, layer_inputs):
    #    if tf.rank(layer_inputs) is 3:
    #        return layer_inputs[:, :, 2]
    #    elif tf.rank(layer_inputs) is 4:
    #        return layer_inputs[:, :, :, 2]
    #    else:
    #        raise Exception('bad input')

    def get_ids_from_inputs(self, layer_inputs, id_type, extra_dim=False):
        id_indices = ['nodes', 'node_labels', 'edge_labels']
        if id_type not in id_indices:
            raise Exception("arg id_type must be one of 'nodes', \
                            'node_labels' or 'edge_labels'")
        id_idx = id_indices.index(id_type)
        if extra_dim:
            return layer_inputs[:, :, :, id_idx]
        else:
            return layer_inputs[:, :, id_idx]

    def get_embeds_with_zeros(self, embeds):
        embed_dim = tf.shape(embeds)[1]
        return tf.concat([embeds, tf.zeros([1, embed_dim])], axis=0)

    def get_node_embeds_from_inputs(self, layer_inputs, node_embeds, extra_dim=False):
        node_ids = self.get_ids_from_inputs(layer_inputs, id_type='nodes', extra_dim=extra_dim)
        embeds_with_zeros = self.get_embeds_with_zeros(node_embeds)
        return tf.nn.embedding_lookup(params=embeds_with_zeros,
                                      ids=node_ids)

    def get_node_label_embeds_from_inputs(self, layer_inputs, extra_dim=False):
        node_label_ids = self.get_ids_from_inputs(layer_inputs, id_type='node_labels', extra_dim=extra_dim)
        embeds_with_zeros = self.get_embeds_with_zeros(self.node_label_embeds)
        return tf.nn.embedding_lookup(params=embeds_with_zeros,
                                      ids=node_label_ids)

    def get_edge_label_embeds(self, layer_inputs, extra_dim=False):
        edge_label_ids = self.get_ids_from_inputs(layer_inputs, id_type='edge_labels', extra_dim=extra_dim)
        embeds_with_zeros = self.get_embeds_with_zeros(self.edge_label_embeds)
        return tf.nn.embedding_lookup(params=embeds_with_zeros,
                                      ids=edge_label_ids)

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

