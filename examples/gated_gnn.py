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
                cell = tf.nn.rnn_cell.GRUCell(net_p.node_embed_size,
                                              activation=self.activation)
            else:
                raise Exception('unsupported rnn cell type')

            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, state_keep_prob=self.node_dropout_keep_prob)

            # create weights for attention @sparse:302

    def __call__(self, placeholders):
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        node_embeds = [placeholders.input_node_embeds] # shape: [V, D]
        num_nodes = tf.shape(placeholders.input_node_embeds)[0]

        message_sources = []  # list of tensors of message sources of shape [E]
        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_label_idx, adj_list_for_edge_label in enumerate(placeholders.adjacency_lists):
            edge_sources = adj_list_for_edge_label[:, 0]
            edge_targets = adj_list_for_edge_label[:, 1]
            message_sources.append(edge_sources)
            message_targets.append(edge_targets)
            message_edge_labels.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_label_idx)
        message_sources = tf.concat(message_sources, axis=0)  # Shape [M]
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        with tf.variable_scope(self.name):
            # TODO: something with residuals, but I don't think we use them anyway
            # TODO: get tensor for the attention for different edge labels

            timestep_node_embeds = [placeholders.input_node_embeds]
            for step in range(self.num_timesteps):
                with tf.variable_scope('timestep_{0}'.format(step)):
                    messages = []  # list of tensors of messages of shape [E, D]
                    message_source_states = []  # list of tensors of edge source states of shape [E, D]

                    for edge_label_idx, adj_list_for_edge_type in enumerate(placeholders.adjacency_lists):
                        edge_sources = adj_list_for_edge_type[:, 0]
                        edge_source_states = tf.nn.embedding_lookup(params=timestep_node_embeds[-1],
                                                                    ids=edge_sources)
                        all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                               self.edge_weights[edge_label_idx])  # Shape [E, D]
                        messages.append(all_messages_for_edge_type)
                        message_source_states.append(edge_source_states)

                    messages = tf.concat(messages, axis=0)  # Shape [M, D]

                    # TODO: do attention on the messages which are being sent to each node

                    incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                                segment_ids=message_targets,
                                                                num_segments=tf.shape(timestep_node_embeds)[0])

                    incoming_messages += tf.matmul(placeholders.num_incoming_edges_per_type,
                                                   self.edge_biases[edge_label_idx])

                    num_incoming_edges = tf.reduce_sum(placeholders.num_incoming_edges_per_label,
                                                       keep_dims=True, axis=-1)  # Shape [V, 1]
                    incoming_messages /= num_incoming_edges + tf.keras.backend.epsilon()

                    incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                     axis=-1) # Shape [V, D]

                    timestep_node_embeds.append(self.rnn_cell(incoming_information,
                                                              timestep_node_embeds[-1]))[1] # Shape [V, D]

        placeholders.input_node_embeds = timestep_node_embeds[-1]
        return placeholders
