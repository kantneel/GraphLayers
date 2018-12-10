import sys
sys.path.append('../../')
import tensorflow as tf
import numpy as np
from framework.model.layer import GraphLayer

class GatedLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 activation='tanh',
                 edge_dropout_keep_prob=0.8,
                 node_dropout_keep_prob=0.8,
                 num_timesteps=1,
                 cell_type='gru',
                 name='gated_layer'):
        super().__init__(layer_params, network_params, name)

        self.activation = eval('tf.nn.{0}'.format(activation))
        self.num_timesteps = num_timesteps
        self.edge_dropout_keep_prob = edge_dropout_keep_prob
        self.node_dropout_keep_prob = node_dropout_keep_prob
        self.cell_type = cell_type
        self.rnn_state = None

    def create_weights(self):
        net_p = self.network_params
        layer_p = self.layer_params
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name):
            edge_weights = tf.Variable(
                initializer([
                    net_p.num_edge_labels * layer_p.node_embed_size,
                    layer_p.node_embed_size]),
                name='{0}_edge_weights'.format(self.name))

            edge_weights = tf.reshape(edge_weights, [net_p.num_edge_labels,
                                                     layer_p.node_embed_size,
                                                     layer_p.node_embed_size])
            self.edge_weights = tf.nn.dropout(edge_weights,
                                              keep_prob=self.edge_dropout_keep_prob)
            self.edge_biases = tf.Variable(np.zeros([net_p.num_edge_labels,
                                                     layer_p.node_embed_size],
                                                    dtype=np.float32),
                                           name='{0}_edge_bias'.format(self.name))

            if self.cell_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(layer_p.node_embed_size,
                                              activation=self.activation)
            else:
                raise Exception('unsupported rnn cell type')

            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, state_keep_prob=self.node_dropout_keep_prob)

            inp_size = self.layer_params.node_embed_size + \
                self.layer_params.node_label_embed_size + \
                self.layer_params.edge_label_embed_size

            self.dense_layer = tf.layers.Dense(
                units=self.layer_params.node_embed_size,
                activation=self.activation,
                name='inp_proj_layer')
            # create weights for attention @sparse:302

    def __call__(self, layer_input_embeds, target_embeds):
        # input: [n, k, d1+d2+d3], [n, d1]

        # [n, k, d1]
        node_embeds = self.get_node_embeds(layer_input_embeds)
        # [n, k]
        edge_labels = tf.argmax(self.get_edge_label_embeds(layer_input_embeds), axis=2)
        # [n, k, d1, d1]
        edge_label_weights = tf.nn.embedding_lookup(params=self.edge_weights,
                                                    ids=edge_labels)
        # [n, k, d1]
        transformed_node_embeds = tf.squeeze(tf.matmul(edge_label_weights,
                                            tf.expand_dims(node_embeds, 3)), 3)
        # [n, d1]
        incoming_messages = tf.reduce_mean(transformed_node_embeds, axis=1)
        # [n, d1]
        output_embeds = self.rnn_cell(incoming_messages, target_embeds)[1]
        return output_embeds

