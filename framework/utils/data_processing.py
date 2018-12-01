import sys
sys.path.append('../../')
import tqdm
import pickle
import collections
import numpy as np
from framework.utils.io import IndexedFileReader, IndexedFileWriter


class DataProcessor(object):
    def __init__(self, path, tie_fwd_bkwd=True, is_training_data=False):
        self.path = path
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

    def load_data(self, path, use_memory=False, use_disk=False):
        reader = IndexedFileReader(path)
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
