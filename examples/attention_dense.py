import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
from framework.model.layer import GraphLayer

class AttentionDenseLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 activation=None,
                 name='attention_dense'):
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
            self.attention_function = tf.layers.Dense(
                units=1,
                activation=None,
                activation='attn')

    def __call__(self, input_messages, input_node_embeds):
        # inputs: [n, k, 3], [n, d1]

        # [n, k, d1]
        node_embeds = self.get_node_embeds_from_inputs(input_messages,
                                                       input_node_embeds)
        # [n, k, 1]
        attention_scores = self.attention_function(node_embeds)
        score_totals = tf.reduce_sum(attention_scores, axis=1)
        normalized_attention = attention_scores / score_totals

        # [n, k, d1]
        attented_embeds = normalized_attention * node_embeds
        return attended_embeds
