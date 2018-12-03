import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
from framework.model.layer import GraphLayer

class PoolingLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 name='pooling_layer'):
        super().__init__(layer_params, network_params, name)

    def __call__(self, placeholders):
        new_inputs = tf.unsorted_segment_sum(data=placeholders.input_node_embeds,
                                             segment_ids=placeholders.graph_nodes_list,
                                             num_segments=placeholders.num_graphs)

        placeholders.input_node_embeds = new_inputs
        return placeholders

