import sys
sys.path.append('../framework/')
import tensorflow as tf
import numpy as np
from layer import GraphLayer

class GatedLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 activation='tanh',
                 edge_dropout_keep_prob=0.8,
                 node_dropout_keep_prob=0.8,
                 num_timesteps=1,
                 cell_type='gru',
                 name='gated_layer'):
        super().__init__(layer_params, network_params, name)

        self.activation = activation
        self.num_timesteps = num_timesteps
        self.edge_dropout_keep_prob = edge_dropout_keep_prob
        self.node_dropout_keep_prob = node_dropout_keep_prob
        self.cell_type = cell_type

        self.create_weights()


    def create_weights(self):
        net_p = self.network_params
        layer_p = self.layer_params

        with tf.variable_scope(self.name):
            edge_weights = tf.Variable(
                utils.glorot_init([
                    net_p.num_edge_labels * layer_p.node_embed_dim,
                    layer_p.node_embed_dim]),
                name='{0}_edge_weights'.format(self.name))

            edge_weights = tf.reshape(edge_weights, [net_p.num_edge_labels,
                                                     layer_p.node_embed_dim,
                                                     layer_p.node_embed_dim])
            self.edge_weights = tf.nn.dropout(edge_weights,
                                              keep_prob=self.edge_dropout_keep_prob)
            self.edge_biases = tf.Variable(np.zeros([layer_p.num_edge_labels,
                                                     net_p.node_embed_dim],
                                                    dtype=np.float32),
                                           name='{0}_edge_bias'.format(self.name))

            if self.cell_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(net_p.node_embed_dim,
                                              activation=self.activation)
            else:
                raise Exception('unsupported rnn cell type')

            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, state_keep_prob=self.node_dropout_keep_prob)

    def compute_final_representations(self, placeholders):
        pass


