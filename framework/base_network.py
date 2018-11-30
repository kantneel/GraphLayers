import json
import pickle
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import tqdm
import yaml

from autopandas.utils.io import IndexedFileReader, IndexedFileWriter
from ggnn.utils import ParamsNamespace
from ggnn.models import utils


class GraphNetwork(object):
    def __init__(self, network_params):
        self.network_params = network_params
        self.layers = []

    def add_layer(self, new_layer):
        self.layers.append(new_layer)
        # check that layer properties are compatible

    def __call__(self, placeholders):
        results = placeholders
        for layer in self.layers:
            results = layer(results)

        return results

    def define_placeholders(self):
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [None], name='target_values')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')


