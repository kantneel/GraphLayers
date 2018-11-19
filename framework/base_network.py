import json
import pickle
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from abc import ABC, abstractmethod

from typing import Iterable

import tqdm
import yaml

from autopandas.utils.io import IndexedFileReader, IndexedFileWriter
from ggnn.utils import ParamsNamespace
from ggnn.models import utils


class BaseGGNN(ABC):
    def __init__(self):
        self.params: ParamsNamespace = ParamsNamespace()
        self.params.update(self.default_params())

    @classmethod
    def default_params(cls):
        return {
            'args': ParamsNamespace(),
            'num_epochs': 3000,
            'patience': 25,
            'lr': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size_node': 100,
            'hidden_size_final_mlp': 100,
            'num_timesteps': 4,

            'tie_fwd_bkwd': True,

            'random_seed': 0,
        }

    @classmethod
    def from_params(cls, params: ParamsNamespace):
        res = cls()
        res.params.update(params)
        return res

        # --------------------------------------------------------------- #
        #  Model Definition
        # --------------------------------------------------------------- #

    def make_model(self, mode):
        self.define_placeholders()

        #  First, compute the node-level representations, after the message-passing algorithm
        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            self.ops['final_node_representations'] = self.compute_final_node_representations()

        with tf.variable_scope("out_layer"):
            # Should return logits with dimension equal to the number of output classes
            logits = self.prepare_final_layer()
            labels = self.placeholders['target_values']

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            self.ops['loss'] = tf.reduce_mean(loss)
            probabilities = tf.nn.softmax(logits)

            correct_prediction = tf.equal(tf.argmax(probabilities, -1), self.placeholders['target_values'])
            self.ops['accuracy_task'] = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            top_k = tf.nn.top_k(probabilities, k=self.params['num_classes'])
            self.ops['preds'] = top_k.indices
            self.ops['probs'] = top_k.values

    def define_placeholders(self):
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [None], name='target_values')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')

    @abstractmethod
    def prepare_specific_graph_model(self):
        pass

    @abstractmethod
    def compute_final_node_representations(self) -> tf.Tensor:
        pass

    def prepare_final_layer(self):
        #  By default, pools up the node-embeddings (sum by default),
        #  and applies a simple MLP
        pooled = self.perform_pooling(self.ops['final_node_representations'])
        self.ops['final_predictor'] = self.final_predictor()
        return self.ops['final_predictor'](pooled)

    def perform_pooling(self, last_h):
        #  By default, it simply sums up the node embeddings
        #  We do not assume sorted segment_ids
        graph_node_sums = tf.unsorted_segment_sum(data=last_h,  # [v x h]
                                                  segment_ids=self.placeholders['graph_nodes_list'],
                                                  num_segments=self.placeholders['num_graphs'])  # [g x h]

        return graph_node_sums

    def final_predictor(self):
        #  By default, a simple MLP with one hidden layer
        return utils.MLP(self.params['hidden_size_node'], self.params['num_classes'],
                         [self.params['hidden_size_final_mlp']],
                         self.placeholders['out_layer_dropout_keep_prob'])

    # --------------------------------------------------------------- #
    #  Model Training Definition
    # --------------------------------------------------------------- #

    def build_graph_model(self, mode='training', restore_file=None):
        possible_modes = ['training', 'testing', 'inference']
        if mode not in possible_modes:
            raise NotImplementedError("Mode has to be one of {}".format(", ".join(possible_modes)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model(mode=mode)
            if mode == 'training':
                self.make_train_step()

            if restore_file is None:
                self.initialize_model()
            else:
                self.restore_model(restore_file)

    def initialize_model(self):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.model.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        debug = self.args.get('debug', False)
        if debug:
            print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    if debug:
                        print("Restoring weights for %s" % variable.name)
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    if debug:
                        print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)

            if debug:
                for var_name in data_to_load['weights']:
                    if var_name not in used_vars:
                        print('Saved weights for %s not used by model.' % var_name)

            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.model.sess.run(restore_ops)

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.model.sess.run(variable)

        data_to_save = {
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.params.args.freeze_graph_model:
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars

        optimizer = tf.train.AdamOptimizer(self.params['lr'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))

        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)

        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    # --------------------------------------------------------------- #
    #  Data Processing
    # --------------------------------------------------------------- #

    @abstractmethod
    def preprocess_data(self, path, is_training_data=False):
        pass

    def load_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)
        if self.params.args.get('use_memory', False):
            result = self.process_raw_graphs(reader, is_training_data)
            reader.close()
            return result

        if self.params.args.get('use_disk', False):
            w = IndexedFileWriter(path + '.processed')
            for d in tqdm.tqdm(reader, desc='Dumping processed graphs to disk'):
                w.append(pickle.dumps(self.process_raw_graph(d)))

            w.close()
            reader.close()
            return IndexedFileReader(path + '.processed')

        #  We won't pre-process anything. We'll convert on-the-fly. Saves memory but is very slow and wasteful
        reader.set_loader(lambda x: self.process_raw_graph(pickle.load(x)))
        return reader

    def process_raw_graphs(self, raw_data: Iterable, is_training_data: bool = False):
        processed_graphs = []
        for d in tqdm.tqdm(raw_data, desc='Processing Raw Data'):
            processed_graphs.append(self.process_raw_graph(d))

        if is_training_data:
            np.random.shuffle(processed_graphs)

        return processed_graphs

    @abstractmethod
    def process_raw_graph(self, graph):
        pass

    @abstractmethod
    def make_minibatch_iterator(self, data, is_training: bool):
        pass

    @abstractmethod
    def save_interface(self, path: str):
        pass

