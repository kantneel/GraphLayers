from abc import ABC, abstractmethod

class GraphLayer(ABC):
    """A GraphLayer is a function that transforms a set of node embeddings"""
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn_cell_impl.py#L174
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/engine/base_layer.py#L71

    def __init__(self, node_embed_dim, node_label_embed_dim,
                 edge_label_embed_dim, output_embed_dim, **kwargs):
        self.node_embed_dim = node_embed_dim
        self.node_label_embed_dim = node_label_embed_dim
        self.edge_label_embed_dim = edge_label_embed_dim
        self.output_embed_dim = output_embed_dim

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
