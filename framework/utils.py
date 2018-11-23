from collections import namedtuple

class GraphNetworkParams(namedtuple('GraphNetworkParams',
                                    ['num_nodes', 'num_node_labels',
                                     'num_edge_labels'])):

    def __init__(self, *args):
        self.check_valid_args(args)
        super().__init__(args)

    def check_valid_args(self, args):
        pass

class GraphLayerParams(namedtuple('GraphLayerParams',
                                  ['node_embed_dim', 'node_label_embed_dim',
                                   'edge_label_embed_dim', 'output_embed_dim'])):

    def __init__(self, *args):
        self.check_valid_args(args)
        super().__init__(args)

    def check_valid_args(self, args):
        pass
