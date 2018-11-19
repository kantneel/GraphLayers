import tqdm
import tensorflow as tf
import numpy as np

from autopandas.utils.io import IndexedFileReader
from ggnn.models.sparse import SparseGGNN
from ggnn.models import utils


class SparseGGNN_RNN(SparseGGNN):
    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'max_beam_trees' : None,
            'use_function_embeddings' : False
        })

        return params

    def preprocess_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)

        num_fwd_edge_types = 0
        annotation_size = 0
        num_classes = 0
        max_depth = 1
        for g in tqdm.tqdm(reader, desc='Preliminary Data Pass'):
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['edges']] + [-1]) + 1)
            annotation_size = max(annotation_size, max(g['node_features']) + 1)
            label = g.get('label_seq', [g.get('label', 0)])
            num_classes = max(num_classes, max(label) + 1)
            max_depth = max(max_depth, len(label))

        self.params['num_edge_types'] = num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)
        self.params['annotation_size'] = annotation_size
        self.params['num_classes'] = num_classes
        self.params['max_depth'] = max_depth
        reader.close()

    def process_raw_graph(self, graph):
        (adjacency_lists, num_incoming_edge_per_type) = self.graph_to_adjacency_lists(graph['edges'])
        label = graph.get('label_seq', [graph.get('label', 0)])
        return {"adjacency_lists": adjacency_lists,
                "num_incoming_edge_per_type": num_incoming_edge_per_type,
                "init": self.to_one_hot(graph["node_features"], self.params['annotation_size']),
                "label": label}

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [None, None], name='target_values')

    def final_predictor(self, pooled):
        #  By default, a simple MLP with one hidden layer
        return utils.LSTMDecoder(self.params['hidden_size_node'], self.params['num_classes'],
                                 self.params['hidden_size_final_mlp'], self.params['max_depth'],
                                 self.placeholders['out_layer_dropout_keep_prob'],
                                 self.params['max_beam_trees'], pooled,
                                 self.params['use_function_embeddings'],
                                 self.placeholders['target_values'])

    def prepare_final_layer(self):
        #  By default, pools up the node-embeddings (sum by default),
        #  and applies a simple MLP
        pooled = self.perform_pooling(self.ops['final_node_representations'])
        self.ops['final_predictor'] = self.final_predictor(pooled)
        return self.ops['final_predictor'](pooled)

    def make_model(self, mode):
        self.define_placeholders()

        #  First, compute the node-level representations, after the message-passing algorithm
        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            self.ops['final_node_representations'] = self.compute_final_node_representations()

        with tf.variable_scope("out_layer"):
            # Should return logits with dimension equal to the number of output classes
            logits = self.prepare_final_layer()
            labels = self.placeholders['target_values']

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            self.ops['loss'] = tf.reduce_mean(loss)
            probabilities = tf.nn.softmax(logits)

            #  Shape of correct_prediction is [max_depth, batch_size]
            correct_prediction = tf.equal(tf.argmax(probabilities, -1), self.placeholders['target_values'])
            correct_prediction = tf.reduce_all(correct_prediction, axis=0)  # shape is [batch_size]
            self.ops['accuracy_task'] = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            top_k = tf.nn.top_k(probabilities, k=self.params['num_classes'])
            self.ops['preds'] = top_k.indices
            self.ops['probs'] = top_k.values

    def make_minibatch_iterator(self, data, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            if isinstance(data, IndexedFileReader):
                data.shuffle()
            else:
                np.random.shuffle(data)

        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0

        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_adjacency_lists = [[] for _ in range(self.params['num_edge_types'])]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         (
                                             (0, 0),
                                             (0, self.params['hidden_size_node'] - self.params['annotation_size'])),
                                         'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(
                    np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.params['num_edge_types']):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.params['num_edge_types']))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                batch_target_task_values.append(cur_graph['label'])
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            #  Only difference from sparse
            #  TODO : What is the following doing?
            batch_target_task_values = [list(l) for l in zip(*batch_target_task_values)]

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values']: batch_target_task_values,
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.params['num_edge_types']):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict
