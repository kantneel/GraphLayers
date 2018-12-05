import sys
sys.path.append('../../')
import tqdm
import pickle
import collections
import numpy as np
import tensorflow as tf
from framework.utils.io import IndexedFileReader, IndexedFileWriter


class DataProcessor(object):
    def __init__(self, path, batch_size, layer_params, tie_fwd_bkwd=True, is_training_data=False):
        self.path = path
        self.batch_size = batch_size
        self.layer_params = layer_params
        self.tie_fwd_bkwd = tie_fwd_bkwd
        self.is_training_data = is_training_data

        reader = IndexedFileReader(path)

        num_fwd_edge_labels = 0
        num_node_labels = 0
        num_classes = 0
        depth = 1
        for g in tqdm.tqdm(reader, desc='Preliminary Data Pass', dynamic_ncols=True):
            num_fwd_edge_labels = max(num_fwd_edge_labels, max([e[1] for e in g['edges']] + [-1]) + 1)
            num_node_labels = max(num_node_labels, max(g['node_features']) + 1)
            num_classes = max(num_classes, g['label'] + 1)

        reader.close()
        self.num_edge_labels = num_fwd_edge_labels * (1 if tie_fwd_bkwd else 2)
        self.num_node_labels = num_node_labels
        self.num_classes = num_classes

        self.placeholders = self.define_placeholders()

    def define_placeholders(self):
        placeholders = {
            'target_values' : tf.placeholder(tf.int64, [None], name='target_values'),
            'num_graphs' : tf.placeholder(tf.int32, [], name='num_graphs'),
            'graph_nodes_list' : tf.placeholder(tf.int32, [None], name='graph_nodes_list'),
            'graph_state_keep_prob' : tf.placeholder(tf.float32, None, name='graph_state_keep_prob'),
            'out_layer_dropout_keep_prob' : tf.placeholder(
                tf.float32, [], name='out_layer_dropout_keep_prob'),
            'input_node_embeds' : tf.placeholder(
                tf.float32, [None, self.layer_params.node_embed_size], name='node_embeds'),
            'node_labels' : tf.placeholder(
                tf.float32, [None], name='node_labels'),
            'adjacency_lists' : [tf.placeholder(
                tf.int32, [None, 2], name='adjacency_e%s' % e) for e in range(self.num_edge_labels)],
            'edge_weight_dropout_keep_prob' : tf.placeholder(
                tf.float32, None, name='edge_weight_dropout_keep_prob')
        }
        return placeholders


    def load_data(self, use_memory=False, use_disk=False):
        reader = IndexedFileReader(self.path)
        if use_memory:
            result = self.process_raw_graphs(reader, self.is_training_data)
            reader.close()
            return result

        if use_disk:
            w = IndexedFileWriter(path + '.processed')
            for d in tqdm.tqdm(reader, desc='Dumping processed graphs to disk'):
                w.append(pickle.dumps(self.process_raw_graph(d)))

            w.close()
            reader.close()
            return IndexedFileReader(path + '.processed')

        #  We won't pre-process anything. We'll convert on-the-fly. Saves memory but is very slow and wasteful
        reader.set_loader(lambda x: self.process_raw_graph(pickle.load(x)))
        return reader

    def process_raw_graphs(self, raw_data):
        processed_graphs = []
        for d in tqdm.tqdm(raw_data, desc='Processing Raw Data'):
            processed_graphs.append(self.process_raw_graph(d, self.num_node_labels))

        if self.is_training_data:
            np.random.shuffle(processed_graphs)

        return processed_graphs

    def process_raw_graph(self, graph):
        (adjacency_lists, num_incoming_edge_per_label) = self.graph_to_adjacency_lists(graph['edges'])
        return {"adjacency_lists": adjacency_lists,
                "num_incoming_edge_per_label": num_incoming_edge_per_label,
                "init": self.to_one_hot(graph["node_features"], self.num_node_labels),
                "label": graph.get("label", 0)}

    def graph_to_adjacency_lists(self, graph):
        adj_lists = collections.defaultdict(list)
        num_incoming_edges_dicts_per_label = {}
        for src, e, dest in graph:
            fwd_edge_label = e
            adj_lists[fwd_edge_label].append((src, dest))
            if fwd_edge_label not in num_incoming_edges_dicts_per_label:
                num_incoming_edges_dicts_per_label[fwd_edge_label] = collections.defaultdict(int)

            num_incoming_edges_dicts_per_label[fwd_edge_label][dest] += 1
            if self.tie_fwd_bkwd:
                adj_lists[fwd_edge_label].append((dest, src))
                num_incoming_edges_dicts_per_label[fwd_edge_label][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not self.tie_fwd_bkwd:
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_label = self.num_edge_labels + edge_label
                final_adj_lists[bwd_edge_label] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                if bwd_edge_label not in num_incoming_edges_dicts_per_label:
                    num_incoming_edges_dicts_per_label[bwd_edge_label] = collections.defaultdict(int)

                for (x, y) in edges:
                    num_incoming_edges_dicts_per_label[bwd_edge_label][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_label

    def to_one_hot(self, vals, depth):
        res = []
        for val in vals:
            v = [0] * depth
            v[val] = 1
            res.append(v)

        return res

    def make_minibatch_iterator(self):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        dataset = self.load_data(use_memory=True)
        if self.is_training_data:
            if isinstance(dataset, IndexedFileReader):
                dataset.shuffle()
            else:
                np.random.shuffle(dataset)

        # Pack until we cannot fit more graphs in the batch
        #state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        #edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        state_dropout_keep_prob = edge_weights_dropout_keep_prob = 1
        num_graphs = 0

        while num_graphs < len(dataset):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_labels)]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(dataset) and node_offset + len(dataset[num_graphs]['init']) < self.batch_size:
                cur_graph = dataset[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         (
                                             (0, 0),
                                             (0, self.layer_params.node_embed_size - self.layer_params.node_label_embed_size)),
                                         'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(
                    np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.num_edge_labels):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_label = np.zeros((num_nodes_in_graph, self.num_edge_labels))
                for (e_type, num_incoming_edges_per_label_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_label_dict.items():
                        num_incoming_edges_per_label[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_label.append(num_incoming_edges_per_label)
                batch_target_task_values.append(cur_graph['label'])
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                self.placeholders['input_node_embeds'] : np.array(batch_node_features),
                self.placeholders['node_labels'] : np.argmax(np.array(batch_node_features), axis=1)
                self.placeholders['graph_nodes_list'] : np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values'] : batch_target_task_values,
                self.placeholders['num_graphs'] : num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob'] : state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob'] : edge_weights_dropout_keep_prob
            }
            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.num_edge_labels):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict
