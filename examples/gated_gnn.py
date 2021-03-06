import sys
sys.path.append('../../')
import tensorflow as tf
import numpy as np
from framework.model.layer import GraphLayer

class GatedLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 activation='tanh',
                 edge_dropout_keep_prob=1,
                 node_dropout_keep_prob=1,
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
        self.input_config.set_config(False, False, True)

    def create_weights(self):
        net_p = self.network_params
        layer_p = self.layer_params

        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name):
            edge_weights = tf.Variable(
                initializer([
                    net_p.num_edge_labels * layer_p.node_embed_size,
                    layer_p.node_embed_size]),
                name='edge_weights')

            edge_weights = tf.reshape(edge_weights, [net_p.num_edge_labels,
                                                     layer_p.node_embed_size,
                                                     layer_p.node_embed_size])
            self.edge_weights = tf.nn.dropout(edge_weights,
                                              keep_prob=self.edge_dropout_keep_prob)
            self.edge_biases = tf.Variable(np.zeros([net_p.num_edge_labels,
                                                     layer_p.node_embed_size],
                                                    dtype=np.float32),
                                           name='edge_bias')

            if self.cell_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(layer_p.node_embed_size,
                                              activation=self.activation)
            elif self.cell_type == 'lstm':
                cell = tf.nn.rnn_cell.LSTMCell(layer_p.node_embed_size,
                                               activation=self.activation)
            else:
                raise Exception('unsupported rnn cell type')

            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, state_keep_prob=self.node_dropout_keep_prob)

    def create_node_label_embeds(self):
        self.create_default_node_label_embeds()

    def create_edge_label_embeds(self):
        self.create_default_edge_label_embeds()

    def __call__(self, layer_input_embeds, target_embeds):
        if self.network_params.use_sparse:
            return self.call_sparse(layer_input_embeds, target_embeds)
        else:
            return self.call_dense(layer_input_embeds, target_embeds)


    def call_dense(self, layer_input_embeds, target_embeds):
        # input: [num_edge_labels, n, k, 3], [n, d1]

        # [num_edge_labels, n, k, d1]
        node_embeds = self.get_node_embeds_from_inputs(layer_input_embeds, target_embeds)

        # self.edge_weights - [num_edge_labels, d1, d1]
        # [num_edge_labels, n, k, d1]
        transformed_node_embeds = tf.einsum('enkd,eod->enko', node_embeds, self.edge_weights)
        mask = 1 - tf.cast(tf.equal(transformed_node_embeds, 0), tf.float32)
        # [num_edge_labels, 1, 1, d1]
        edge_biases = tf.expand_dims(tf.expand_dims(self.edge_biases, 1), 1)
        edge_biases *= mask

        transformed_node_embeds += edge_biases
        # [n, k, d1]
        transformed_node_embeds = tf.reduce_sum(transformed_node_embeds, axis=0)

        #num_nonzero = tf.reduce_sum(tf.cast(transformed_node_embeds[:, :, 0], tf.bool), axis=1)
        num_nonzero = tf.cast(tf.count_nonzero(transformed_node_embeds[:, :, 0], axis=1, keep_dims=True), tf.float32)
        # [n, d1]
        # change this to divide by the number nonzero rather than reduce mean
        incoming_messages = tf.reduce_sum(transformed_node_embeds, axis=1) / num_nonzero
        # [n, d1]
        output_embeds = self.rnn_cell(incoming_messages, target_embeds)[1]
        return output_embeds

    def call_sparse(self, layer_input_embeds, target_embeds):
        # input: [num_edge_labels, n, k, 3], [n, d1]

        # [num_edge_labels, n, k, d1]
        node_embeds = self.get_node_embeds_from_inputs(
            layer_input_embeds, target_embeds)

        split_node_embeds = tf.sparse_split(
            sp_input=node_embeds,
            num_split=self.network_params.num_edge_labels,
            axis=0)

        transformed_sparse_messages = []
        for edge_idx, split_embed in enumerate(split_node_embeds):
            reshaped_embed = tf.sparse_reshape(
                sp_input=split_embed,
                shape=[-1, self.layer_params.node_embed_size])

            transformed_embeds = tf.sparse.matmul(
                sp_a=reshaped_embed,
                b=self.edge_weights[edge_idx])


            indices = tf.cast(tf.where(tf.not_equal(transformed_embeds, 0)), tf.int64)
            values = tf.gather_nd(transformed_embeds, indices)
            dense_shape = tf.shape(transformed_embeds, out_type=tf.int64)

            sparse_transformed = tf.SparseTensor(indices=indices, values=values,
                                                 dense_shape=dense_shape)

            #edge_bias = self.edge_biases[edge_idx] + 1e-4
            #tiled_edge_bias = tf.tile(edge_bias, [tf.round(tf.size(values) / tf.size(edge_bias))])

            #sparse_bias = tf.SparseTensor(indices=indices, values=tiled_edge_bias,
            #                              dense_shape=dense_shape)

            #sparse_transformed = tf.sparse_add(
            #    sparse_transformed, sparse_bias, thresh=1e-6)

            sparse_transformed_reshaped = tf.sparse_reshape(
                sp_input=sparse_transformed,
                shape=split_embed.dense_shape)

            transformed_sparse_messages.append(sparse_transformed_reshaped)

        concat_sparse_messages = tf.sparse_concat(
            axis=0, sp_inputs=transformed_sparse_messages)

        summed_to_get_num_incoming_edges = tf.sparse.reduce_sum(
            sp_input=concat_sparse_messages, axis=[0, 3])

        num_nonzero = tf.cast(tf.reduce_any(
            tf.abs(summed_to_get_num_incoming_edges) > 1e-6, axis=1, keepdims=True), tf.float32)
        # [n, d1]
        # change this to divide by the number nonzero rather than reduce mean
        incoming_messages = tf.sparse.reduce_sum(
            sp_input=concat_sparse_messages, axis=[0, 2]) / num_nonzero

        incoming_messages = tf.reshape(incoming_messages, [-1, self.layer_params.node_embed_size])
        # [n, d1]
        output_embeds = self.rnn_cell(incoming_messages, target_embeds)[1]
        return output_embeds

