
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

        self.layer_list = []

    def _check_valid_args(self):
        # a bunch of assertions to make sure shapes make sense
        # and are of valid types and that the optional arguments come
        # in appropriate pairs

    def add_layer(self):
        pass
