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
        inp_size = self.layer_params.node_embed_size + \
            self.layer_params.node_label_embed_size + \
            self.layer_params.edge_label_embed_size
        with tf.variable_scope(self.name):
            self.dense_layer = tf.layers.Dense(
                units=self.layer_params.node_embed_size,
                activation=self.activation,
                name='layer')

    def __call__(self, concat_embeds, message_targets):
        transformed_messages = self.dense_layer(concat_embeds)
        new_node_embeds = tf.unsorted_segment_sum(
            data=transformed_messages,
            segment_ids=message_targets,
            num_segments=tf.reduce_max(message_targets) + 1)

        return new_node_embeds
