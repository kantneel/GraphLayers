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

    def _get_ids_from_inputs(self, layer_inputs, id_type):
        """
        Returns a slice of an id tensor produced by
        GraphNetwork().get_messages().
        """
        is_sparse = tf.keras.backend.is_sparse(layer_inputs)
        if is_sparse:
            reordered_inputs = tf.sparse.reorder(layer_inputs)
            dense_layer_inputs = tf.sparse.to_dense(reordered_inputs)
        else:
            dense_layer_inputs = layer_inputs

        id_indices = ['nodes', 'node_labels', 'edge_labels']
        if id_type not in id_indices:
            raise Exception("arg id_type must be one of 'nodes', \
                            'node_labels' or 'edge_labels'")
        id_idx = id_indices.index(id_type)
        rank = len(dense_layer_inputs.get_shape().as_list())
        if rank == 4:
            dense_slice = dense_layer_inputs[:, :, :, id_idx]
        elif rank == 3:
            dense_slice = dense_layer_inputs[:, :, id_idx]
        else:
            raise Exception('dense_layer_inputs has invalid rank')

        if is_sparse:
            indices = tf.cast(tf.where(
                tf.not_equal(dense_slice, 0)), tf.int64)
            values = tf.gather_nd(dense_slice, indices)
            dense_shape = tf.shape(dense_slice, out_type=tf.int64)

            return_ids = tf.SparseTensor(
                indices=indices, values=values, dense_shape=dense_shape)
        else:
            return_ids = dense_slice

        return return_ids

    def _get_embeds_from_ids(self, ids, embeds):
        """
        Fetch the correct embeddings from embeds, depending on whether
        ids is a dense or sparse tensor
        """
        is_sparse = tf.keras.backend.is_sparse(ids)
        if is_sparse:
            # [m, d]
            fetched_embeds = tf.nn.embedding_lookup(
                params=embeds, ids=ids.values)
            embed_dim = tf.shape(fetched_embeds)[1]
            # [m * d]
            reshaped_embeds = tf.reshape(fetched_embeds, [-1])
            # [m * d, 2]
            tiled_indices = tf.contrib.seq2seq.tile_batch(
                ids.indices, embed_dim)

            embed_range = tf.range(embed_dim)
            # [m * d, 1]
            tiled_embed_range = tf.tile(
                embed_range, [tf.shape(ids.indices)[0]])
            tiled_embed_range = tf.expand_dims(tiled_embed_range, 1)
            # [m * d, 3]
            new_sparse_indices = tf.concat(
                [tf.cast(tiled_indices, tf.int32), tiled_embed_range], axis=1)

            rank = len(ids.shape.as_list())
            shape = tf.shape(ids)
            new_dense_shape = [shape[0], shape[1], embed_dim]
            if rank == 3:
                new_dense_shape.insert(2, shape[2])

            return_embeds = tf.SparseTensor(
                indices=tf.cast(new_sparse_indices, tf.int64),
                values=reshaped_embeds,
                dense_shape=new_dense_shape)
        else:
            embed_dim = tf.shape(embeds)[1]
            embeds_with_zeros = tf.concat(
                [embeds, tf.zeros([1, embed_dim])], axis=0)
            return_embeds = tf.nn.embedding_lookup(
                params=embeds_with_zeros, ids=ids)

        return return_embeds

    ##################################################################################
    # Getting embeddings from an id tensor returned by GraphNetwork().get_messages() #
    # First, the method will get ids by slicing the layer_inputs tensor and then     #
    # it uses a modified embedding lookup becasuse of filler ids.                    #
    ##################################################################################

    def get_node_embeds_from_inputs(self, layer_inputs, input_node_embeds):
        node_ids = self._get_ids_from_inputs(
            layer_inputs, id_type='nodes')
        node_embeds = self._get_embeds_from_ids(
            node_ids, input_node_embeds)
        return node_embeds

    def get_node_label_embeds_from_inputs(self, layer_inputs):
        node_label_ids = self._get_ids_from_inputs(
            layer_inputs, id_type='node_labels')
        node_label_embeds = self._get_embeds_from_ids(
            node_label_ids, self.node_label_embeds)
        return node_label_embeds

    def get_edge_label_embeds_from_inputs(self, layer_inputs):
        edge_label_ids = self._get_ids_from_inputs(
            layer_inputs, id_type='edge_labels')
        edge_label_embeds = self._get_embeds_from_ids(
            edge_label_ids, self.edge_label_embeds)
        return edge_label_embeds

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

