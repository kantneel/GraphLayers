import sys
sys.path.append('../framework/')
from layer import GraphLayer

class GatedLayer(GraphLayer):
    """Layer from Gated Graph Sequence Neural Networks"""

    def __init__(self, num_nodes, node_embed_size,
                 adj_matrix, num_timesteps,
                 node_type_ids=None, node_type_embeds=None,
                 edge_type_ids=None, edge_type_embeds=None):
        """
        Args - all are tf.tensors/ops/placeholders or None:

        - node_type_ids ([num_nodes]):
            what types each of these nodes are (optional)
        - node_type_embeds ([num_node_types x node_type_embed_size]):
            the embeddings of each node type (optional)
        - edge_type_ids ([num_nodes x num_nodes]):
            what types each of these edges are (optional)
        - edge_type_embeds ([num_edge_types x edge_type_embed_size]):
            the embeddings of each edge type (optional)
        """

        super().__init__(num_nodes, node_embed_size, adj_matrix)
        self.num_timesteps = num_timesteps
        self.node_type_ids = node_type_ids
        self.node_type_embeds = node_type_embeds
        self.edge_type_ids = edge_type_ids
        self.edge_type_embeds = edge_type_embeds

        self._check_valid_args()
        self.sparse_adj = False
        if isinstance(self.adj_matrix, tf.SparseTensor):
            self.sparse_adj = True

        self.set_node_type_attrs(node_type_ids, node_type_embeds)
        self.set_edge_type_attrs(edge_type_ids, edge_type_embeds)

    def set_node_type_attrs(self, ids, embeds):
        """Takes tf.tensors/ops/placeholders and assigns the values"""
        self.has_node_types = False
        if ids is not None:
            # TODO: check ids shape and embeds shape
            self.has_node_types = True
            self.node_type_ids = ids
            self.node_type_embeds = embeds

            # the number of node types this layer can use
            self.num_node_types = tf.shape(self.node_type_embeds)[0]
            # the dimension of node type embeds
            self.node_type_embed_size = tf.shape(self.node_type_embeds)[1]

    def set_edge_type_attrs(self, ids, embeds):
        """Takes tf.tensors/ops/placeholders and assigns the values"""
        self.has_edge_types = False
        if ids is not None:
            # TODO: check ids shape and embeds shape
            self.has_edge_types = True
            self.edge_type_ids = ids
            self.edge_type_embeds = embeds

            # the number of edge types this layer can use
            self.num_edge_types = tf.shape(self.edge_type_embeds)[0]
            # the dimension of edge type embeds
            self.edge_type_embed_size = tf.shape(self.edge_type_embeds)[1]

    def _check_valid_args(self):
        """
        a bunch of assertions to make sure shapes make sense
        and are of valid types and that the optional arguments come
        in appropriate pairs
        """
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __call__(self, node_embeds):
        """Act on node_embeds"""
        pass
