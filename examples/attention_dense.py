import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
from examples.basic_dense import BasicDenseLayer

class AttentionDenseLayer(BasicDenseLayer):
    def __init__(self, layer_params, network_params,
                 activation=None,
                 name='attention_dense'):
        super().__init__(layer_params, network_params,
                         activation, name)

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

    def __call__(self, input_messages, current_node_embeds):
        # inputs: [n, k, 3], [n, d1]

        node_embeds = self.get_node_embeds_from_inputs(
            input_messages, current_node_embeds)

        if self.network_params.use_sparse:
            node_embeds = tf.sparse_to_dense(node_embeds)

        # [n, k, 1]
        attention_scores = self.attention_function(node_embeds)
        score_totals = tf.reduce_sum(
            attention_scores, axis=1)
        normalized_attention = attention_scores / score_totals

        # [n, k, d1]
        attented_embeds = normalized_attention * node_embeds
        return attended_embeds
