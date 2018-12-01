from collections import namedtuple

class GraphNetworkParams(object):
    def __init__(self, num_nodes, num_node_labels=1, num_edge_labels=1):
        self.check_valid_args(args)
        self.num_nodes = num_nodes
        self.num_node_labels = num_node_labels
        self.num_edge_labels = num_edge_labels

    def check_valid_args(self, args):
        pass

class GraphLayerParams(object):
    def __init__(self, node_embed_dim, node_label_embed_dim=None,
                edge_label_embed_dim=None, output_embed_dim=None):
        self.check_valid_args(args)
        self.node_embed_dim = node_embed_dim
        self.node_label_embed_dim = node_label_embed_dim
        self.edge_label_embed_dim = edge_label_embed_dim
        if output_embed_dim is None:
            self.output_embed_dim = self.node_embed_dim
        else:
            self.output_embed_dim = output_embed_dim

    def check_valid_args(self, args):
        pass

class GraphLayerPlaceholders(object):
    def __init__(self, input_node_embeds, adjacency_lists,
                 num_incoming_edges_per_label, graph_nodes_list):
        self.check_valid_args(args)
        self.input_node_embeds = input_node_embeds
        self.adjacency_lists = adjacency_lists
        self.num_incoming_edges_per_label = num_incoming_edges_per_label
        self.graph_nodes_list = graph_nodes_list

    def check_valid_args(self, args):
        pass
