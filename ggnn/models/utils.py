#!/usr/bin/env/python

import numpy as np
import tensorflow as tf
import queue
import threading
from ggnn.models import beam_search

SMALL_NUMBER = 1e-7


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int = 2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    @staticmethod
    def init_weights(shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden


class LSTMDecoder(object):
    def __init__(self, in_size, out_size, hid_size, depth, dropout_keep_prob,
                 max_beam_trees, input_encoding, use_function_embeddings, labels):
        """A recurrent decoder which outputs label probabilities/ids for a given graph embedding input"""
        # parameters for building decoder
        self.in_size = in_size
        self.out_size = out_size
        self.hid_size = out_size # compatibility for teacher forcing
        self.depth = depth
        self.dropout_keep_prob = dropout_keep_prob

        # options for customizing decoder structure
        self.use_function_embeddings = use_function_embeddings
        self.max_beam_trees = max_beam_trees

        # tensor valued data needed for graph building
        self.batch_size = tf.shape(input_encoding)[0]
        self.input_encoding = tf.layers.dense(input_encoding, self.hid_size, name='inp_proj')
        self.labels = labels

        # Variables and modules
        self.fn_embedding_layer = self.create_fn_embedding_layer()
        self.decoder_cell = self.create_decoder_cell()
        self.training_decoder = self.create_training_decoder()
        self.beam_search_decoder = self.create_beam_search_decoder()


    @staticmethod
    def init_weights(shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def create_decoder_cell(self):
        """Make a gated recurrent cell to be used in training_decoder and beam_search_decoder"""
        num_proj = self.out_size if self.max_beam_trees is None else None
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hid_size, num_proj=num_proj,
                                       name='LSTM_cell', dtype=tf.float32)

        return tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=self.dropout_keep_prob,
                                             output_keep_prob=self.dropout_keep_prob,
                                             state_keep_prob=self.dropout_keep_prob)

    def create_fn_embedding_layer(self):
        """Create variable for learning function label embeddings"""
        embedding_layer = None
        if self.use_function_embeddings:
            embedding_layer = tf.Variable(self.init_weights((self.out_size, self.hid_size)),
                                          name='fn_embeds')

        return embedding_layer

    def create_training_decoder(self):
        """Create teacher-forcing decoder to be used at training time"""
        start_input = tf.zeros([1, self.batch_size, self.out_size])
        teacher_force_labels = tf.slice(self.labels, [0, 0], [self.depth - 1, -1])
        teacher_force_label_inp = tf.nn.embedding_lookup(tf.eye(self.out_size), teacher_force_labels)

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.concat([start_input, teacher_force_label_inp], 0),
            sequence_length=tf.fill([self.batch_size], self.depth),
            time_major=True)

        zero_c, zero_h = tf.contrib.rnn.BasicLSTMCell(self.hid_size).zero_state(self.batch_size, tf.float32)
        # initialize state with the input encoding, first inputs will be START_TOKEN
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(zero_c, self.input_encoding)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            helper=training_helper,
            initial_state=initial_state,
            output_layer=tf.layers.Dense(self.out_size))

        return training_decoder

    def create_beam_search_decoder(self):
        """Create beam search decoder to perhaps be used at test time"""
        beam_search_decoder = None
        if self.max_beam_trees is not None:

            # Need to get true values of these
            START_TOKEN = 0
            END_TOKEN = 1

            zero_c, zero_h = tf.contrib.rnn.BasicLSTMCell(self.hid_size).zero_state(self.batch_size, tf.float32)
            # initialize state with the input encoding, first inputs will be START_TOKEN
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(zero_c, self.input_encoding)
            # create beam_trees copies of each item in the batch
            tiled_initial_state = tf.contrib.seq2seq.tile_batch(initial_state, self.max_beam_trees)

            beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=tf.eye(self.out_size), # one-hot encoded
                start_tokens=tf.fill([self.batch_size], START_TOKEN),
                end_token=END_TOKEN,
                initial_state=tiled_initial_state,
                beam_width=self.max_beam_trees,
                output_layer=tf.layers.Dense(self.out_size))

        return beam_search_decoder

    def __call__(self, inputs, maybe_beam_search=False):
        """Use the LSTMDecoder on inputs"""
        if maybe_beam_search and self.max_beam_trees is not None:
            beam_outputs, final_state, final_seq_lens = \
                tf.contrib.seq2seq.dynamic_decode(self.beam_search_decoder,
                                                  output_time_major=True,
                                                  maximum_iterations=self.depth)
            # permute into shape [beam x time x batch]
            # in order to easily check top_k accuracy via each index in 0th dimension
            decoder_out = tf.transpose(beam_outputs, perm=[2, 0, 1])
        else:
            training_outputs, final_state, final_seq_lens = \
                tf.contrib.seq2seq.dynamic_decode(self.training_decoder,
                                                  output_time_major=True,
                                                  impute_finished=True,
                                                  maximum_iterations=self.depth)
            decoder_out = training_outputs.rnn_output

        if self.use_function_embeddings:
            decoder_out = tf.tensordot(decoder_out, self.fn_embedding_layer, [[2], [1]])

        return decoder_out


