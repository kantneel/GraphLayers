from abc import ABC, abstractmethod

class GraphLayer(ABC):
    """A GraphLayer is a function that transforms a set of node embeddings"""
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn_cell_impl.py#L174
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/engine/base_layer.py#L71

    def __init__(self, num_nodes, node_embed_size, adj_matrix, **kwargs):
        """
        Args - all are tf.tensors/ops/placeholders

        - num_nodes: number of nodes this layer handles
        - node_embed_size: dimension of node embeddings
        - adj_matrix ([num_nodes x num_nodes]):
            which nodes are connected to one another in directed edges
        """

        """
        node label embed dim
        edge label embed dim
        node embed dim
        """
        self.num_nodes = num_nodes
        self.node_embed_size = node_embed_size
        self.adj_matrix = adj_matrix
        self.out_shape = self.compute_output_shape()

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Abstract Method")

    def compute_output_shape(self):
        """
        Find out what the output shape is. This is important for
        building a GraphLayerList, and also knowing what the output
        shape is of an entire GraphNetwork.
        """
        pass
