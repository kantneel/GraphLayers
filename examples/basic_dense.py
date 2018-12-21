import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
from framework.model.layer import GraphLayer

class BasicDenseLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 activation=None,
                 name='basic_dense'):
        super().__init__(layer_params, network_params, name)

        self.activation = activation
        if activation is not None:
            self.activation = eval('tf.nn.{0}'.format(activation))

    def create_weights(self):
        with tf.variable_scope(self.name):
            self.dense_layer = tf.layers.Dense(
                units=self.layer_params.node_embed_size,
                activation=self.activation,
                name='layer')

    def create_node_label_embeds(self):
        self.create_default_node_label_embeds()

    def create_edge_label_embeds(self):
        self.create_default_edge_label_embeds()

    def __call__(self, input_messages, current_node_embeds):
        node_embeds = self.get_node_embeds_from_inputs(
            input_messages, current_node_embeds)

        if self.network_params.use_sparse:
            node_embeds = tf.sparse.to_dense(node_embeds)
        summed_messages = tf.reduce_sum(
            node_embeds, axis=1)

        transformed_messages = self.dense_layer(summed_messages)
        return transformed_messages
