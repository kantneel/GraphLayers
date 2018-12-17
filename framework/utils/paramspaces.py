class GraphNetworkParams(object):
    def __init__(self, num_nodes, num_node_labels=1, num_edge_labels=1):
        self.num_nodes = num_nodes
        self.num_node_labels = num_node_labels
        self.num_edge_labels = num_edge_labels

    def check_valid_args(self, args):
        pass


class GraphLayerParams(object):
    def __init__(self, node_embed_size, node_label_embed_size=None,
                edge_label_embed_size=None, output_embed_size=None):
        self.node_embed_size = node_embed_size
        self.node_label_embed_size = node_label_embed_size
        self.edge_label_embed_size = edge_label_embed_size
        if output_embed_size is None:
            self.output_embed_size = self.node_embed_size
        else:
            self.output_embed_size = output_embed_size

    def check_valid_args(self, args):
        pass


class ExperimentParams(object):
    def __init__(self,
                 lr=0.001,
                 num_epochs=500,
                 patience=25,
                 analysis=False,
                 label_mapping=None,
                 wdir=None,
                 top_k=1,
                 mode='train',
                 model=None,
                 train=None,
                 valid=None,
                 clamp_grad_norm=1,
                 rng=2018):
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.analysis = analysis
        self.label_mapping = label_mapping
        self.wdir = wdir
        self.top_k = top_k
        self.mode = mode
        self.model = model
        self.train = train
        self.valid = valid
        self.clamp_grad_norm = clamp_grad_norm
        self.rng = rng

    def check_valid_args(self, args):
        pass


class GraphNetworkPlaceholders(object):
    def __init__(self, input_node_embeds, num_graphs, sorted_messages,
                 in_degree_indices, node_labels, graph_nodes_list, targets):

        # to be used at every typical layer
        self.input_node_embeds = input_node_embeds
        self.node_labels = node_labels
        self.in_degree_indices = in_degree_indices,
        self.sorted_messages = sorted_messages

        # to be used by only some layers or the output
        self.num_graphs = num_graphs
        self.graph_nodes_list = graph_nodes_list
        self.targets = targets

    def check_valid_args(self, args):
        pass

class InputConfig(object):
    def __init__(self, source_only=True,
                 source_node_labels=False,
                 edge_labels=False):
        self.source_only = source_only
        self.source_node_labels = source_node_labels
        self.edge_labels = edge_labels

    def set_config(self, source_only, source_node_labels, edge_labels):
        assert True
        self.source_only = source_only
        self.source_node_labels = source_node_labels
        self.edge_labels = edge_labels

    @classmethod
    def default(cls):
        return cls()


