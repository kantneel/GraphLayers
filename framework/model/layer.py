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
        """Create a copy of a layer to conveniently add multiple times."""
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

    def _get_ids_from_inputs(self, layer_inputs, id_type, extra_dim=False):
        """
        Returns a slice of an id tensor produced by
        GraphNetwork().get_messages().
        extra_dim refers to whether the id tensor has rank 3 or 4.
        """

        id_indices = ['nodes', 'node_labels', 'edge_labels']
        if id_type not in id_indices:
            raise Exception("arg id_type must be one of 'nodes', \
                            'node_labels' or 'edge_labels'")
        id_idx = id_indices.index(id_type)
        if extra_dim:
            return layer_inputs[:, :, :, id_idx]
        else:
            return layer_inputs[:, :, id_idx]

    def _get_embeds_with_zeros(self, embeds):
        """
        The embeddings need an extra row which is all zeros. This is because
        the id tensors produced by GraphNetwork().get_messages() have filler
        rows which have id equal to the total number of ids of that type.
        When those are used as ids for looking up embeddings, they will thus
        be zero.
        """
        embed_dim = tf.shape(embeds)[1]
        return tf.concat([embeds, tf.zeros([1, embed_dim])], axis=0)

    ##################################################################################
    # Getting embeddings from an id tensor returned by GraphNetwork().get_messages() #
    # First, the method will get ids by slicing the layer_inputs tensor and then     #
    # it uses a modified embedding lookup becasuse of filler ids.                    #
    ##################################################################################

    def get_node_embeds_from_inputs(self, layer_inputs, node_embeds, extra_dim=False):
        node_ids = self._get_ids_from_inputs(layer_inputs, id_type='nodes', extra_dim=extra_dim)
        embeds_with_zeros = self._get_embeds_with_zeros(node_embeds)
        return tf.nn.embedding_lookup(params=embeds_with_zeros,
                                      ids=node_ids)

    def get_node_label_embeds_from_inputs(self, layer_inputs, extra_dim=False):
        node_label_ids = self._get_ids_from_inputs(layer_inputs, id_type='node_labels', extra_dim=extra_dim)
        embeds_with_zeros = self._get_embeds_with_zeros(self.node_label_embeds)
        return tf.nn.embedding_lookup(params=embeds_with_zeros,
                                      ids=node_label_ids)

    def get_edge_label_embeds_from_inputs(self, layer_inputs, extra_dim=False):
        edge_label_ids = self._get_ids_from_inputs(layer_inputs, id_type='edge_labels', extra_dim=extra_dim)
        embeds_with_zeros = self._get_embeds_with_zeros(self.edge_label_embeds)
        return tf.nn.embedding_lookup(params=embeds_with_zeros,
                                      ids=edge_label_ids)


    ##################################################################################
    # Node label and edge label embeddings are attributes of a layer, not the entire #
    # network, so they are required. Here are some sensible example embeddings.      #
    ##################################################################################

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
        """
        How a layer takes in a set of inputs and produces a new set of
        node embeddings. In essence, this operation takes in a tensor
        of shape [num_nodes, max_degree, 3] and returns a tensor
        of shape [num_nodes, node_embed_size].
        """
        raise NotImplementedError("Abstract Method")

    def __str__(self):
        pass

    def __repr__(self):
        pass

