import sys
sys.path.append('../../')
import tqdm
import pickle
import collections
import numpy as np
import tensorflow as tf
from framework.utils.io import IndexedFileReader, IndexedFileWriter

class DataProcessor(object):
    def __init__(self, path, batch_size, node_embed_size, tie_fwd_bkwd=True,
                 is_training_data=False, run_prelim_pass=True):
        self.path = path
        self.batch_size = batch_size
        self.node_embed_size = node_embed_size
        self.tie_fwd_bkwd = tie_fwd_bkwd
        self.is_training_data = is_training_data

        if run_prelim_pass:
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

    @classmethod
    def copy_params(cls, other, path, is_training_data):
        assert isinstance(other, DataProcessor)
        instance = cls(path, other.batch_size, other.node_embed_size,
                       other.tie_fwd_bkwd, is_training_data, run_prelim_pass=False)
        instance.num_edge_labels = other.num_edge_labels
        instance.num_node_labels = other.num_node_labels
        instance.num_classes = other.num_classes

        return instance

    def load_data(self, use_memory=False, use_disk=False):
        reader = IndexedFileReader(self.path)
        if use_memory:
            processed_graphs = []
            for d in tqdm.tqdm(reader, desc='Processing Raw Data'):
                processed_graphs.append(self.process_raw_graph(d))
            reader.close()
            return processed_graphs

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

    def process_raw_graph(self, graph):
        adj_lists = collections.defaultdict(list)
        for src, e, dest in graph['edges']:
            fwd_edge_label = e
            adj_lists[fwd_edge_label].append((src, dest))
            if self.tie_fwd_bkwd:
                adj_lists[fwd_edge_label].append((dest, src))

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not self.tie_fwd_bkwd:
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_label = self.num_edge_labels + edge_label
                final_adj_lists[bwd_edge_label] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)

        return {"adjacency_lists": final_adj_lists,
                "init": self.to_one_hot(graph["node_features"], self.num_node_labels),
                "label": graph.get("label", 0)}

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
        dataset = self.load_data()
        if self.is_training_data:
            if isinstance(dataset, IndexedFileReader):
                dataset.shuffle()
            else:
                np.random.shuffle(dataset)

        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(dataset):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_labels)]
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(dataset) and node_offset + len(dataset[num_graphs]['init']) < self.batch_size:
                cur_graph = dataset[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         (
                                             (0, 0),
                                             (0, self.node_embed_size - self.num_node_labels)),
                                         'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(
                    np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.num_edge_labels):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                batch_target_task_values.append(cur_graph['label'])
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph
            node_labels = np.argmax(np.array(batch_node_features), axis=1)
            # Merge adjacency lists and information about incoming nodes:
            in_degrees = [0 for _ in range(len(batch_node_features))]
            all_messages = []
            for i in range(self.num_edge_labels):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                edge_sources = adj_list[:, 0]
                message_node_labels = np.expand_dims(node_labels[edge_sources], 1)
                message_edge_labels = np.expand_dims(np.ones_like(edge_sources, dtype=np.int32) * i, 1)
                messages_of_edge_label = np.concatenate([adj_list, # 0 - source, 1 - target
                                                         message_node_labels, # 2 - source node label
                                                         message_edge_labels], 1) # 3 - edge label
                all_messages.append(messages_of_edge_label)
                for row in adj_list:
                    in_degrees[row[1]] += 1

            concat_messages = np.concatenate(all_messages, 0)
            sorted_messages = concat_messages[np.argsort(concat_messages[:, 1])]

            in_degree_indices = np.zeros((sum(in_degrees), 2))
            message_num = 0
            for i, d in enumerate(in_degrees):
                in_degree_indices[message_num : message_num + d] = \
                    np.vstack([np.ones(d, dtype=int) * i, np.arange(d, dtype=int)]).transpose()
                message_num += d

            batch_feed_dict = {
                'input_node_embeds' : np.array(batch_node_features),
                'node_labels' : node_labels,
                'graph_nodes_list' : np.concatenate(batch_graph_nodes_list),
                'targets' : batch_target_task_values,
                'num_graphs' : num_graphs_in_batch,
                'in_degree_indices' : in_degree_indices,
                'sorted_messages' : sorted_messages
            }
            yield batch_feed_dict
