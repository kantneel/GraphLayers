from abc import ABC, abstractmethod

class GraphLayer(ABC):
    """A GraphLayer is a function that transforms a set of node embeddings"""

    def __init__(self, current_embeddings, adj_matrix,
                 node_type_ids=None, node_type_embeds=None,
                 edge_type_ids=None, edge_type_embeds=None):
        """
        Args - all are tf.tensors/ops/placeholders or None:

        - current_embeddings ([num_nodes x node_embed_size])
            the embeddings [e_1, ..., e_k] which are input to this layer
        - adj_matrix ([num_nodes x num_nodes])
            which nodes are connected to one another in directed edges
        - node_type_ids ([num_nodes])
            what types each of these nodes are (optional)
        - node_type_embeds ([num_node_types x node_type_embed_size])
            the embeddings of each node type (optional)
        - edge_type_ids ([num_nodes x num_nodes])
            what types each of these edges are (optional)
        - edge_type_embeds ([num_edge_types x edge_type_embed_size])
            the embeddings of each edge type (optional)
        """

        self.current_embeddings = current_embeddings
        self.adj_matrix = adj_matrix
        self.node_type_ids = node_type_ids
        self.node_type_embeds = node_type_embeds
        self.edge_type_ids = edge_type_ids
        self.edge_type_embeds = edge_type_embeds

        self._check_valid_args()
        self.sparse_adj = False
        if isinstance(self.adj_matrix, tf.SparseTensor):
            self.sparse_adj = True

        # the number of nodes that this layer handles
        self.num_nodes = tf.shape(current_embeddings)[0]
        # the dimension of the input embeddings
        self.inp_embed_size = tf.shape(current_embeddings)[1]

        if node_type_embeds is not None:
            # the number of node types this layer can use
            self.num_node_types = tf.shape(self.node_type_embeds)[0]
            # the dimension of node type embeds
            self.node_type_embed_size = tf.shape(self.node_type_embeds)[1]
        else:
            self.num_node_types = self.node_type_embed_size = None

        if edge_type_embeds is not None:
            # the number of edge types this layer can use
            self.num_edge_types = tf.shape(self.edge_type_embeds)[0]
            # the dimension of edge type embeds
            self.edge_type_embed_size = tf.shape(self.edge_type_embeds)[1]
        else:
            self.num_edge_types = self.edge_type_embed_size = None

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

    @abstractmethod
    def compute_outputs(self):
        raise NotImplementedError()
