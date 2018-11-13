from layer_list import GraphLayerList
from layer import GraphLayer

class GraphNetwork(object):
    """
    A GraphNetwork is composed of GraphLayers but also handles processing
    which does not depend on the specific GraphLayers.
    """

    def __init__(self, num_nodes, node_embed_size,
                 num_node_types=None, node_type_embed_size=None,
                 num_edge_types=None, edge_type_embed_size=None):
        """
        Args - all are ints or tf.tensors/placeholders or None:

        - num_nodes: the number of nodes this graph network can process in
            one call. Masking will be taken care of.
        - node_embed_size: nodes will necessarily have embeddings
        - num_node_types: number of types of nodes (optional)
        - node_type_embed_size: size of embeddings for node types (optional)
        - num_edge_types: number of types of edges (optional)
        - edge_type_embed_size: size of embeddings for edge types (optional)

        * All of the embed sizes are for the inputs. Conceivably, the sizes
        could change from layer to layer.
        """

        self.num_nodes = num_nodes
        self.node_embed_size = node_embed_size
        self.num_node_types = num_node_types
        self.node_type_embed_size = node_type_embed_size
        self.num_edge_types = num_edge_types
        self.edge_type_embed_size = edge_type_embed_size

        self._check_valid_args()

        self.layer_list = GraphLayerList()

    def _check_valid_args(self):
        """
        A bunch of assertions to make sure types make sense
        and that the optional arguments come in appropriate pairs
        """
        pass

    def set_layer_list(self, layer_list):
        assert isinstance(layer_list, GraphLayerList)
        self.layer_list = layer_list


    def add_layer(self, graph_layer):
        self.layer_list.append(graph_layer)

    def set_layer_list(self, graph_layer_list):
        self.layer_list = graph_layer_list

    def __str__(self):
        pass

    def __repr__(self):
        pass
