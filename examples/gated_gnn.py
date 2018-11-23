import sys
sys.path.append('../framework/')
from layer import GraphLayer


class GatedLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 activation='tanh',
                 edge_dropout_keep_prob=0.8,
                 node_dropout_keep_prob=0.8
                 cell_type='gru',
                 name='gated_layer'):
        super().__init__(layer_params, network_params)

        self.activation = activation
        self.edge_dropout_keep_prob = edge_dropout_keep_prob
        self.node_dropout_keep_prob = node_dropout_keep_prob
        self.cell_type = cell_type
        self.name = name

        self.create_weights()


    def create_weights(self):
        with tf.variable_scope(self.name):
            edge_weights = tf.Variable(
                utils.glorot_init([
                    self.network_params.num_edge_labels * self.layer_params.node_embed_dim,
                    self.layer_params.node_embed_dim]),
                name='{0}_edge_weights'.format(self.name))

            edge_weights = tf.reshape(edge_weights, [self.network_params.num_edge_labels,
                                                     self.layer_params.node_embed_dim,
                                                     self.layer_params.node_embed_dim])
            self.edge_weights = tf.nn.dropout(edge_weights, keep_prob=self.edge_dropout_keep_prob)
            self.edge_biases = tf.Variable(np.zeros([self.layer_params.num_edge_labels,
                                                     self.network_params.node_embed_dim],
                                                    dtype=np.float32),
                                           name='{0}_edge_bias'.format(self.name))

            if self.cell_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(self.network_params.node_embed_dim,
                                              activation=self.activation)
            else:
                raise Exception('unsupported rnn cell type')

            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, state_keep_prob=self.node_dropout_keep_prob)




