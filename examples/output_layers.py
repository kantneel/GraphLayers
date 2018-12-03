import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
from framework.model.layer import GraphLayer

class DenseOutputLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 hidden_units_list,
                 num_classes,
                 activation='relu',
                 name='dense_output'):

        super().__init__(layer_params, network_params, name)
        self.hidden_units_list = hidden_units_list
        self.num_classes = num_classes
        self.activation = eval('tf.nn.{0}'.format(activation))

        self.create_weights()

    def create_weights(self):
        inp_size = self.layer_params.node_embed_size
        self.dense_layers = []
        with tf.variable_scope(self.name):
            for i, hidden_size in enumerate(self.hidden_units_list):
                self.dense_layers.append(
                    tf.layers.Dense(hidden_size, self.activation,
                                    name='layer_{0}'.format(i)))

            # producing logits / regression values
            self.dense_layers.append(
                tf.layers.Dense(self.num_classes, name='final'))

    def __call__(self, placeholders):
        # might be graph embeds but it's okay
        outputs = [placeholders.input_node_embeds]
        for layer in self.dense_layers:
            outputs.append(layer(outputs[-1]))

        placeholders.input_node_embeds = outputs[-1]
        return placeholders

class SoftmaxLossLayer(GraphLayer):
    def __init__(self, layer_params, network_params,
                 name='softmax_loss'):
        super().__init__(layer_params, network_params, name)

    def __call__(self, placeholders):
        pass

